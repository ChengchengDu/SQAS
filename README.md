# SQAS
the code for paper "Query-focused Abstractive Summarization via Question-answering Model"
## SQAS
本文是利用通用摘要数据集进行QFS任务的初步尝试，并利用问答任务解决编码解码框架中缺乏推理决策过程的问题， 其总体过程如下：
### 查询摘要数据集构造
将CNN/DailyMail和XSUM分别转换成两个QFS数据集，分别为CNNDM_Q和XSUM_Q。
```
cd SQAS/data_utils
bash summary_squad_utils.sh
```
### 模型训练
模型包括三个模块，分别是问答模块，摘要生成模块和互信息模块。
其中问答模块捕捉细粒度信息，提高模型的推理能力和查询建模能力。
摘要生成模块用于生成和查询相关的摘要内容。
互信息模块用于构建问答模块和摘要模块的之间的桥梁，在摘要生成的过程融入更多细粒度信息。
```
cd SQAS/train_utils
bash train_one_cnndm.sh
```
### 模型评测
测试数据是在原始数据集进行改造的，其构造方式如下：首先，本文根据原始参考摘要获取查询语句，然后使用查询语句对源文档进行检索，即根据指标进行贪婪搜索。最终选择最多三个句子作为 QFS 的目标摘要，并将其从原文中删除。
```
cd SQAS/train_utils
bash run_cnndm_eval.sh
```
