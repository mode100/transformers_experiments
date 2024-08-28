# このプロジェクトについて
HuggingFace Transformersを使って、Transformerの実験を行っています。

# 実験結果
### 1. princess タスク
プリンセスが言いそうな文章を生成するタスク。手動で作った40個程度の文章を用いてファインチューニングした。
##### 生成できた文章：
```
私も、いつか、一緒に行きますわ。
明日はお天気が良くなりそうですわ。
最近は、もっぱら「海」にハマっていますわ。
なんか最近、暑くて死にたくなりますわ。
```

### 2. dog cat タスク
dogとcatのみで構成された10語の文章が与えられ、catのまとまりが何個あるか数えるタスク。例えば、「cat cat cat dog cat」は2に対応する。言語モデルをファインチューニングした結果、正解率100%を達成した。Transformerによる言語モデルはこの数学的な操作を表現できることが分かった。
