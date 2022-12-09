안녕하세요
USAD (UnSupervised AutoEncoder) 알고리즘을 제 멋대로 개량하여
Train 환경과 다른 제한된 Test Features를 가지고도 추론할 수 있게 지식증류 방법론을 적용한 Code 입니다.
그런데, 현 상태로써는 성능이 나오지 않습니다 ㅠㅠ
그러나 아이디어를 공유 드리면 남은 대회기간 동안 여러분께
조그만 Insight 라도 드릴 수 있지 않을까 싶어 코드 공유 올립니다. (하단 Github repo. 주소)

Project 구조:
    - /data
    - /result                             모델 가중치가 저장된 곳
    - model.py                       AutoEncoder 모델이 정의된 객체
    - utils.py                           기타 함수
    - DataLoader.py           Raw data를 network Input으로 만들기 위한 객체
    - train_teacher.py        Teacher Model 학습 Code (Full-dataset으로 AutoEncoder 학습)
    - train_student.py        Student Model 학습 Code (지식증류적용, Teacher의 Output을 복원 대상으로 하여 Student 학습)

# Student 학습 시 정답 값으로 Teacher의 LatentVector도 모방하도록 설계하려 했으나 시간 문제 상 어려울 것으로 판단하여 중단했습니다

글쓴이의 공유 코드 Github repo.
https://github.com/K-imlab/KDUAD

reference
    참고 논문: Audibert, J., et al. (2020, August). Usad: Unsupervised anomaly detection on multivariate time series.

