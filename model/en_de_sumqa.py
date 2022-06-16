from typing import List, Optional, Tuple
import bisect
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.bart_model import BartModel, PretrainedBartModel, BartQuestionAnsweringHead
from model.bart_model import _make_linear_from_emb, shift_tokens_right
from model.loss import label_smoothed_nll_loss
from model.output import EncDecQaMimBartOutput
import math


def _get_best_indexes(start_logits, end_logits, n_best_size, sentence_start_positions, sentence_end_positions):
    """Given sentence_start_positions or sentence_end_positions to get n-best logits from a logits list
    找到当前feature 抽取到的摘要的开始位置和结束位置"""

    assert len(sentence_start_positions) == len(sentence_end_positions)
    n_best_size = n_best_size if n_best_size < len(sentence_start_positions) else len(sentence_end_positions)
    sentence_start_positions_logits = start_logits[sentence_start_positions]
    sentence_end_positions_logtis = end_logits[sentence_end_positions]
    logits = sentence_start_positions_logits + sentence_end_positions_logtis  # 因为我们选择的是一个句子 因此start_logtis和end_logits是成对出现的

    # 对当前的logits进行排序
    index_and_scores = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True
    )

    index_and_scores = index_and_scores[: n_best_size]

    # 抽取到的句子的开始和结束位置
    extract_start_positions = []
    extract_end_positions = []
    _start_logits = []
    _end_logits = []
    for index_and_score in index_and_scores:
        pos_index = index_and_score[0]
        extract_start_positions.append(sentence_start_positions[pos_index])
        extract_end_positions.append(sentence_end_positions[pos_index])
        _start_logits.append(start_logits[sentence_start_positions[pos_index]])
        _end_logits.append(end_logits[sentence_end_positions[pos_index]])   # 找到对应位置的logits

    return extract_start_positions, extract_end_positions, _start_logits, _end_logits


class Summary_hiddens(nn.Module):
    def __init__(self):
        super(Summary_hiddens, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor):
        assert tensor.dim() == 2 or tensor.dim() == 3
        if tensor.dim() == 3:
            return self.sigmoid(torch.mean(tensor, dim=1))   # batch_size, seq_len, hidden_dim -> batch_size, hidden_dim
        else:
            return self.sigmoid(torch.mean(tensor, dim=0))    # seq_len, hidden_dim -> hidden_dim


def trim_tensor(input):
    """抽取不为-1的前面的元素"""
    keep_column_mask = input.ne(-1)
    input = input[keep_column_mask]
    return input


# encoder端互信息表征的抽取
class ExtractHiddens(nn.Module):
    def __init__(self):
        super(ExtractHiddens, self).__init__()
        self.summary = Summary_hiddens()

    def forward(self,
                encoder_hiddens,
                decoder_hiddens,
                start_positions,    # 是补全之后的标签  不是只包含答案位置的tensor
                end_positions,
                sum_start_positions=None,
                sum_end_positions=None,
                context_len=3
                ):
        """
            获取start-positions 和end positions之间的表征信息作为抽取到的摘要信息 然后将其进行总结得到表征向量
            :param hiddens: batch_size, seq_len, hidden_dim
            :param start_logits:
            :param end_logits:
            :param n_best_size:
            :return: 针对整个batch输入返回抽取到的摘要的平均表征 1, som_len, hidden_dim->1, 1, hidden_dim
            需要增加一个相似度的计算  根据相似度找到所要计算互信息的摘要句子
            """

        batch_size, encoder_seq_len, hidden_size = encoder_hiddens.shape

        # 获取batch中每一个答案的上下文
        extract_answers_enc = []    # encoder端抽取到的摘要的总结性的表示    每一个元素都是以batch_size seq_len, hidden_dim表示
        generation_summaries_dec = []   # 整体摘要表示
        generation_tokens_summaries_dec = []   # decoder生成的摘要的每一个token的表示 没有进行总结的
        for batch_id in range(batch_size):
            start_pos = start_positions[batch_id]  # 因为抽到的句子可能有多个， 因此需要将每一个句子都进行拼接
            end_pos = end_positions[batch_id]      # 位置标签

            start_pos = start_pos.int()
            end_pos = end_pos.int()
            true_start_pos = start_pos - context_len if (start_pos - context_len) >= 0 else 0
            true_end_pos = end_pos + context_len if(end_pos + context_len) <= encoder_seq_len else encoder_seq_len
            # # 如果计算互信息的时候包含当前的句子的话 就不需要进行判断 下面的条件
            extract_answer_enc = encoder_hiddens[batch_id, true_start_pos: true_end_pos + 1, :]  # 抽取到的张量是二维的 抽到的是句子的表示
            # 整个batch个example的抽取式摘要的表示
            extract_answers_enc.append(extract_answer_enc.unsqueeze(0))  # 扩充一个batch的维度 batch_size, encoder_seq_len, hidden_dim 可能会做总结

            # 针对decoder的表征进行
            # decoder-encoder互信息
            # 当前生成的摘要句子的总结性表征和encoder端抽取到的摘要token的互信息
            cur_generation_summary = decoder_hiddens[batch_id, :, :]
            generation_tokens_summaries_dec.append(cur_generation_summary.unsqueeze(0))  # 扩充到batch , seq_len, hidden_size

            # 根据与答案span相似度 选择最相似的摘要句子得到互信息
            extract_answer_enc_summary = self.summary(extract_answer_enc)
            best_start_pos, best_end_pos = 0, 0
            best_similarity = float("inf")
            sum_start_poses = sum_start_positions[batch_id]
            sum_end_poses = sum_end_positions[batch_id]

            for sum_start_pos, sum_end_pos in zip(sum_start_poses, sum_end_poses):
                if sum_start_pos == -1 or sum_end_pos == -1:
                    continue
                cur_sent_summary = cur_generation_summary[sum_start_pos: sum_end_pos,:]
                cur_sent_summary = self.summary(cur_sent_summary)
                cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
                similarity = cos(cur_sent_summary, extract_answer_enc_summary)   # 均是一维的
                if similarity < best_similarity:
                    best_similarity = similarity
                    best_start_pos = sum_start_pos
                    best_end_pos = sum_end_pos

            cur_generation_summary = cur_generation_summary[best_start_pos: best_end_pos + 1]
            cur_generation_summary = self.summary(cur_generation_summary)      # 相当于是一个tokend的表征

            generation_summaries_dec.append(cur_generation_summary.unsqueeze(0))   # 每个元素是 (1,hidden_size)

        extract_answers_enc_fake = [extract_answers_enc[-1]] + extract_answers_enc[:-1]
        generation_summaries_dec_fake = [generation_summaries_dec[-1]] + generation_summaries_dec[:-1]   # 用于计算decoder端的当前生成的宅啊哟表示和encoder端抽取到的token表示的互信息
        generation_tokens_summaries_dec_fake = [generation_tokens_summaries_dec[-1]] + generation_tokens_summaries_dec[:-1]

        return {
            "extract_answers_enc": extract_answers_enc,
            "generation_summaries_dec": generation_summaries_dec,
            "extract_answers_enc_fake": extract_answers_enc_fake,
            "generation_summaries_dec_fake": generation_summaries_dec_fake,
            "generation_tokens_summaries_dec": generation_tokens_summaries_dec,
            "generation_tokens_summaries_dec_fake": generation_tokens_summaries_dec_fake
        }


# 原论文中的discriminator

class Discriminator(nn.Module):
    def __init__(self, d_model):
        super(Discriminator, self).__init__()
        self.bilinear = nn.Bilinear(d_model, d_model, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, a_enc, b_enc):
        # 相当于是计算所有的token和local的序列的长度进行
        # 计算a_enc和b_enc之间的互信息分数(相关性)
        a_enc = a_enc.unsqueeze(1)  # (batch, 1, global_size)   global_size 表示的是hidden dim
        a_enc = a_enc.expand(-1, b_enc.size(1), -1)  # (batch, encoder_seq_len, global_size)
        # (batch, seq_len, global_size) * (batch, seq_len, local_size) -> (batch, seq_len, 1)
        # 得到的是每一个位置的分数

        scores = self.bilinear(a_enc.contiguous(),
                               b_enc.contiguous())  # 得到每一个token的互信息  分数是不需要进行归一化  在计算损失的时候直接进行归一化

        return scores


class MIMLoss(nn.Module):
    '''
    Deep infomax loss for SQuAD question answering dataset.
    As the difference between GC and LC only lies in whether we do summarization over x,
    this class can be used as both GC and LC.
    '''

    def __init__(self, feature_size):
        super(MIMLoss, self).__init__()
        self.discriminator = Discriminator(feature_size)
        self.summarize = Summary_hiddens()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.dropout = nn.Dropout(0.1)

    def forward(self, x_enc, x_fake, y_enc, y_fake, do_summarize=False):
        '''
        Args:
            global_enc, local_enc: (batch, seq, dim)  只不过batch_size=1
            global_fake, local_fake: (batch, seq, dim)
        '''
        # Compute g(x, y)
        # 如果不进行摘要 那么只是随机选择一个token的表示  输入时的张量维度就是 (1， hidden_dim)
        if do_summarize:
            x_enc = self.summarize(x_enc)  # 对答案得到总结性的表示  batch_size, hidden_dim  2维的对三维的互信息
        x_enc = self.dropout(x_enc)
        y_enc = self.dropout(y_enc)  # batch_size, dim   当前的batch_size实际上是1
        logits = self.discriminator(x_enc, y_enc)  # x_enc 表示答案  y_enc表示上下文信息   discriminator 实际上就是g函数
        batch_size1, n_seq1 = y_enc.size(0), y_enc.size(1)
        labels = torch.ones(batch_size1, n_seq1)
        # Compute 1 - g(x, y^(\bar))
        y_fake = self.dropout(y_fake)
        _logits = self.discriminator(x_enc, y_fake)

        batch_size2, n_seq2 = y_fake.size(0), y_fake.size(1)
        _labels = torch.zeros(batch_size2, n_seq2)  # 0代表的就是假的

        logits, labels = torch.cat((logits, _logits), dim=1), torch.cat((labels, _labels), dim=1)
        # Compute 1 - g(x^(\bar), y)
        if do_summarize:
            x_fake = self.summarize(x_fake)
        x_fake = self.dropout(x_fake)
        _logits = self.discriminator(x_fake, y_enc)
        _labels = torch.zeros(batch_size1, n_seq1)

        # 相当于判断所有的序列生成的是真的还是假的
        logits, labels = torch.cat((logits, _logits), dim=1), torch.cat((labels, _labels), dim=1)

        loss = self.bce_loss(logits.squeeze(2), labels.cuda())  # bce的两个输入的shape是一样的

        return loss


# debug检查
class EncDecQaMimBart(PretrainedBartModel):
    """
    encoder-decoder-question-answering-mutual-information-maximization bart
    encoder端的问答损失
    encoder端的互信息 (decoder端的互信息 decoder端的互信息去掉了)
    decoder端的摘要损失

    """

    def __init__(self, config):
        super(EncDecQaMimBart, self).__init__(config)
        base_model = BartModel(config)
        self.model = base_model
        self.pad_token_id = config.pad_token_id

        self.qa_outputs = nn.Linear(config.d_model, 2)
        self.model._init_weights(self.qa_outputs)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.get_hiddens = ExtractHiddens()  # 获取encoder或者是decoder的表征
        # 这里应该定义两个互信息 而不是一个
        self.mim_loss = MIMLoss(config.d_model)

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """扩充词表的大小"""
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens, old_num_tokens):
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            start_positions=None,
            end_positions=None,
            sum_start_positions=None,
            sum_end_positions=None,
            label_smoothing=0,
            return_dict=True,
            **unused,
    ):
        """一共包含三个损失 encoder端的问答损失 decoder端的摘要损失 以及en-decoder端的互信息损失(这一部分的损失暂时不计
        将encoder的hiddens和decoder进行拼接计算decoder端的输出)"""

        # 获取decoder_input_ids
        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.model.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # Encoder端的损失
        encoder_hiddens = outputs.encoder_last_hidden_state
        decoder_hiddens = outputs[0]   # seq2seq的last_hidden_states
        batch_size = encoder_hiddens.size(0)

        total_loss = None

        # 2) 问答的答案只有一个
        qa_loss = None
        qa_logits = self.qa_outputs(encoder_hiddens)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)    # shape:(batch_size, )   # 这里怎么能unsqueeze呢？
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)  # clamp_函数positions_labels的长度控制在序列的长度范围上  超过的位置全部使用seq_len代替
            end_positions.clamp_(0, ignored_index)
            qa_loss_fun = nn.CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = qa_loss_fun(start_logits, start_positions)
            end_loss = qa_loss_fun(end_logits, end_positions)

            qa_loss = (start_loss + end_loss) / 2

        #  计算摘要的损失
        summary_loss = None
        lm_logits = F.linear(decoder_hiddens, self.model.shared.weight, bias=self.final_logits_bias)
        if labels is not None:
            if label_smoothing == 0:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
                assert lm_logits.shape[-1] == self.model.config.vocab_size
                summary_loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))

            else:
                lprobs = F.log_softmax(lm_logits, dim=-1)
                summary_loss, summary_nll_loss = label_smoothed_nll_loss(
                    lprobs, labels, label_smoothing, ignore_index=self.config.pad_token_id
                )

        # # 计算互信息的损失
        # # 这里面的答案的位置是真实的位置
        # # 这里的互信息只是计算开始的位置和结束位置的上下文的互信息  context_span的长度选择3
        if start_positions is not None and end_positions is not None:
            hiddens_dict = self.get_hiddens(
                encoder_hiddens=encoder_hiddens,
                decoder_hiddens=decoder_hiddens,
                start_positions=start_positions,
                end_positions=end_positions,
                sum_start_positions=sum_start_positions,
                sum_end_positions=sum_end_positions,
                context_len=3
            )  # 注意下start_positions维度是一维的
        #
            extract_answers_enc = hiddens_dict["extract_answers_enc"]    # 抽取到的encoder端的摘要总体表示
            generation_summaries_dec = hiddens_dict["generation_summaries_dec"]    # decoder端的摘要句子表示
            extract_answers_enc_fake = hiddens_dict["extract_answers_enc_fake"]
            generation_summaries_dec_fake = hiddens_dict["generation_summaries_dec_fake"]

            ende_mim_loss = torch.tensor(0).long().to(input_ids)
            mim_loss = None
            for batch_id in range(batch_size):
                # 得到encoder端抽取到的摘要表示 (1, seq_len, hidden_dim)
                extract_answer_enc = extract_answers_enc[batch_id]
                extract_answer_enc_fake = extract_answers_enc_fake[batch_id]

                # decoder端生成的摘要总结表示
                generation_summary_dec = generation_summaries_dec[batch_id]
                generation_summary_dec_fake = generation_summaries_dec_fake[batch_id]

                # 计算encoder-decoder端的互信息  和抽取式摘要的tok级别的互信息
                ende_mim_loss = ende_mim_loss + self.mim_loss(
                    generation_summary_dec,
                    generation_summary_dec_fake,
                    extract_answer_enc,
                    extract_answer_enc_fake
                )

            mim_loss = ende_mim_loss / batch_size

        # # mim_loss = en_de_mim_tok_loss + en_de_mim_sen_loss
        total_loss = qa_loss + summary_loss + 3 * mim_loss   # qa_loss的系数不需要乘后面会趋向稳定 mim_loss*10  但是xsum没有

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((total_loss,) + output) if total_loss is None else output
        return EncDecQaMimBartOutput(
            loss=total_loss,
            qa_loss=qa_loss,
            logits=lm_logits,
            summary_loss=summary_loss,
            mim_loss=mim_loss,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=encoder_hiddens,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.model.shared)


def convert_zero_position_to_integer(start_positions, end_positions):
    """将0-1标签转换成位置张量进行计算交叉熵损失 即将标签进行拆分"""
    assert start_positions.size() == end_positions.size()
    batch_size = start_positions.size(0)
    # 构建三个标签进行交叉熵损失叠加
    max_labels = 3  # 当前任务的最大标签数是3
    starts = torch.zeros(batch_size, max_labels).fill_(-1)
    ends = torch.zeros(batch_size, max_labels).fill_(-1)
    for batch_id in range(batch_size):
        cur_start_positions = torch.nonzero(start_positions[batch_id], as_tuple=False).squeeze(1)
        cur_end_positions = torch.nonzero(end_positions[batch_id], as_tuple=False).squeeze(1)
        labels_len = cur_start_positions.size(0)
        starts.data[batch_id, :labels_len] = cur_start_positions.data
        ends.data[batch_id, :labels_len] = cur_end_positions.data

    return starts, ends








        


