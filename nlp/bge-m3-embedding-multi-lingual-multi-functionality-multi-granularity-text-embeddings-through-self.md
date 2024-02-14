# BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self



### Overview

* BAAI라는 중국 AI 연구소에서 만든 embedding model
* M3는 다음과 같은 3가지 특징임
  * Multi-Linguality : 100개 이상의 언어
  * Multi-Functionality: 3가지 retrieval 방식을 같이 제공
  * Multi-Granularity : 짧고, 긴 문장(최대 8192 토큰) 에서도 잘 작동함
* 실제 실험결과 한국어에서도 좋은 성능
* 아직 Training코드는 공개되지 않음.

### Introduction

* IR(Information Retrieval)에 사용되는 Embedding 모델은 많이 연구가 됨.&#x20;
* 하지만 다음과 같은 한계점을 가지고 있음
  * 1\) embedding 모델은 대부분 영어에서만 작동
  * 2\) 오직 1개 retrieval task에만 맞춰 학습이 진행됨. (하지만 실제에선 여러개 사용해야할수도 있음)
  * 3\) long-document retreiver 거의 없음
* 다음과 같은 문제를 해결하기 위해서 M3-Embedding을 제안함

<figure><img src="../.gitbook/assets/스크린샷 2024-02-14 오전 8.25.25.png" alt=""><figcaption></figcaption></figure>

* Multi-Linguality
  * 100개 이상 언어를 지원
  * 또한 한국어로 되어있는 문서중에 영어로 질문해서 검색도 가능
* Multi-Functionality
  * 원래 다 각각 학습되고, 추후에 같이 hybrid 방식으로 사용했던 검색 방식을 다 사용할 수 있음.
  * 각각 검색방식은 background에서 자세히 설명
  * Self-knowledge distillation 방식으로 각 3가지 함수에서 나온 score를 통합해서 활용함
* Multi-Granularity
  * 최대 8192 토큰까지 늘림, 이를 위해 batching strategy를 최적화 함
  * 또한 문장, 문단 단위에서 모두 성능이 좋음

### Background&#x20;

#### Dense Retrieval

<figure><img src="../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

* 다음과 같이 사전훈련된 Encoder(Bert, Roberta)를 가지고 임베딩을 활용해서 유사도를 구함
* 이때 각 질문과 문단의 \[CLS] 토큰의 임베딩(hidden state) 값 사용

#### Sparse Retrieval

* 각 단어(token) Term 자체에 집중하는 방법
* 딥러닝 사용전에는 BM25를 가장 많이 사용
  *

      <figure><img src="../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>


* 인코더 모델을 활용한다면 다음과 같이 사용될 수 있음.
  * 각각 토큰의 임베딩(hidden state)를 활용하여 구하는 방법

<figure><img src="../.gitbook/assets/image (2).png" alt=""><figcaption></figcaption></figure>

#### Multi-Vec Retrieval

*   크게 두가지 갈래로 나눠볼수 있음

    * Dense vector와 다르게 \[CLS] 토큰이 아닌 모든 토큰 임베딩을 합쳐서 활용하는것



    <figure><img src="../.gitbook/assets/image (3).png" alt=""><figcaption></figcaption></figure>

    * 다양하게 passage, 질문을 변화시켜 vector를 여러개 만들어서 활용하는 것
    * 질문: 유사질문 만들기로 해서 질문 임베딩 평균 구해서 활용
    * passage: 문서요약, 짧은 문장 등등을 해서 passage 임베딩 평균 구해서 활용 등등
    * 밑에 예시) HyDe

<figure><img src="../.gitbook/assets/image (4).png" alt=""><figcaption></figcaption></figure>

### Method(M3-Embedding)

* Query q가 주어졌을 때 Corpus에서 가장 관련있는 문서 d를 찾아오는 것
  * 이때 q와 D의 언어는 달라도 가능

#### Data Curation

<figure><img src="../.gitbook/assets/스크린샷 2024-02-14 오전 9.19.13.png" alt=""><figcaption></figcaption></figure>

* MultiLongDoc는 직접 생성한 데이터 셋으로 'GPT3.5' 활용
  * You are a curious AI assistant, please generate one specific and valuable question based on the following text. The generated question should revolve around the core content of this text, and avoid using pronouns (e.g., ”this”). Note that you should generate only one question, without including additional content:”.

#### Hybrid Retrieval

* 다음 3가지의 score를 합쳐서 더 좋은 score를 얻는다 이게 기본 포인트
* Dense
  * CLS 토큰의 hidden state를 normalized 한 값 활용
    * $$e_q = norm(Hq[0])$$
    * $$e_p = norm(Hp[0])$$
  * 유사도 스코어는 내적 활용
    * $$s_{dense} = <e_p, e_q>$$
* Sparce(Lexical)
  * 각 토큰의 weight 값 활용
    * $$w_{q_t} = Relu(W^t_{lex}H_q[i])$$
    * $$W_{lex}$$ 는 hidden state를 float로 변환시키는 mapping matrix
  * 유사도 스코어는 joint importance of the co-existed terms를 활용
  *   $$s_{lex} = \sum_{t \in q \cup p} (w_{q_t} * w_{p_t})$$


* Multi-vector
  * Dense vector의 extension으로 전체 output embedding을 활용함
    * $$E_q = norm(W_{mul}^TH_q)$$
    * $$W_{mult}^T$$ 는 leranable projection matrix
  * 유사도는 다음과 같이 구함 (내적 활용)
    *   $$1/N \sum_{i=1}^N max_{j=1}^M E_q[i] \cdot E_p^t[i]$$



#### Self-Knowledge Distillation

* 아까 다른 방식으로 얻은 스코어를 단순히 sum-up 방식으로 합침
  * $$s_{inter} = s_{dense} + s_{lex} + s_{mul}$$
* 그 후 로스값도 이 3가지를 합쳐서 만들어서 학습에 활용 (여기에 기본 로스값인 InfoNCE loss를 같이 사용)
  *   $$L^" = L_{dencse} + L_{lex} + L_{mul}$$


* 학습은 크게 두단계로 진행
  * 1단계 Unsupervised data로 pre-trained
  * 2단계 Supervised data로 앞선 3가지를 loss로 활용해서 훈련

<figure><img src="../.gitbook/assets/스크린샷 2024-02-14 오전 9.33.28.png" alt=""><figcaption></figcaption></figure>

#### Efficient Batching

<figure><img src="../.gitbook/assets/스크린샷 2024-02-14 오전 9.35.07.png" alt="" width="563"><figcaption></figcaption></figure>

<div align="center">

<figure><img src="../.gitbook/assets/스크린샷 2024-02-14 오전 9.36.02.png" alt="" width="563"><figcaption></figcaption></figure>

</div>



### Result

#### Main Result

<figure><img src="../.gitbook/assets/스크린샷 2024-02-14 오전 9.36.49.png" alt=""><figcaption></figcaption></figure>

#### Ablation Study

<figure><img src="../.gitbook/assets/스크린샷 2024-02-14 오전 9.37.55.png" alt=""><figcaption></figcaption></figure>

