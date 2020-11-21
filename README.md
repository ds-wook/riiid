# Riiid
Riiid! Answer Correctness Prediction

## Data
null data -> -1로 제공
```
train.csv
    - row_id: ID code
    - timestamp: user Id
    - Content_id: user interaction의 content id
    - Content_type_id:
        0 if event == question
        1 if event == lecture
    - Task_container_id: question 또는 lecture 모음에 대한 id (ex. 토익의 유형 질문의 묶음)
    - User_answer: user의 답
    - Answer_correctly: target 맞추면 1 아니면 0
    - Prior_question_elapsed_time: 이전 질문에서 답을 내릴 때 소요
    - Prior_question_had_explanation: 이전 질문 이후 답지를 봤는지 여부, 첫번째 interaction은 null

Question.csv
    - Question_id: content_type == question일때, train, test의 content_id와 매칭 되는 id
    - Bundle_id: question이 같이 제공되는지 확인하는 id
    - Correct_answer: 질문의 답
    - Part: 토익의 part
    - Tags: question에 달리는 tag, 내용은 알 수 없음.

Lectures.csv
    - Lecture_id: content_type이 1 일때 train/test의 content_id와 매칭되는 id
    - part: lecture의 파트
    - Tag: lecture에 달리는 tag, 내용은 알 수 없음.
    - Type_of: lecture의 핵심 목적
```
## Strategies
    - Feature engineering 기반 shallow learning
    - 이전에 치뤘던 history 기반해서 피처 생성
        - 이전에 문제 맞춘 이력 기반 피쳐
        - Answer Accuracy history
    - 어떤 question들이 많이 나왔는지
        - Frequency
    - Question을 기준으로 만드는 feature
        - Question이 가지는 통계값들
            - A Question 난이도, frequency등
    - Lecture를 기준으로 만드는 feature
    - Time 기반
        - 이전 question 대비 시간이 얼마나 지났는가 등
    - 다양한 아키텍쳐를 사용한 deep learning (seq2seq, transformer, lstm..)
## cross validation strategies
    - 타임 시리즈일 경우 어떻게 validation하나?
        - 지식은 축적되므로, time 순서로 하는 것이 아무래도 시간 패턴을 유지하면서 학습하여 안전 할 것이다.
    - 실제는 이렇지 않다.
    - User 마다 시작 시간은 상대적으로 정해져 있다. 절대 시작 시간이 없음
    - record 길이도 제각각
## Issue
    - 데이터가 너무 크다.
        - 빠르게 읽고, 처리를 해야 한다.
    - GPU 기반으로!!
        - Lightgbm(gpu built-in)
        - XGBoost
        - CatBoost
        - Rapids AI
## Tips
1. Shallow model과 deep learning 모델을 섞으면 시너지가 크다.
2. 각을 잘 재야 한다.
3. Discussion과 notebook 계속 봐야한다.
4. Ensemble is Key
    - Rank-based average
5. Keep going
