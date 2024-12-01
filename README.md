
# 발표 개요 

1. 알파폴드3는 무엇인가?
	1. 딥마인드에서 개발한 단백질 구조 예측 모델 (딥마인드 사진)
	2. 2024 노벨 화학상  (노벨화학상 사진)
	3. 기존의 도메인 지식을 뛰어넘은 딥러닝 (베이커 교수팀 이긴 자료, CASP 대회 1등 자료 ) 
2. 단백질 구조를 예측하는 이유는? 
	1. 단백질 구성요소가 아닌 3차원 구조에 따라 기능이 결정됨. 같은 시퀀스여도 모양이 다르면 다른 기능을 한다. (열쇠, 기질 특이성 자료)
	2. 단백질 활용: 신약 개발. (신약 개발이 주기가 감소한 자료) 
3. 알파폴드 1,2,3의 변경점 
	1. 1 -> 2: 
		1. 표현 학습 모델 변경: CNN 기반 ResNet ->  attention 기반 Evoformer
		2. End-to-end 학습
	2. 2 -> 3: 
		1. 입력 종류 다양해짐: 하나의 시퀀스로 단일 단백질 구조 예측-> 여러 종류의 입력을 통해 단백질 복합체 구조 예측
		2. 3차원 구조 예측 모델 변경: ResNet -> Diffusion
4. 알파폴드 3가 단백질 구조를 예측하는 방법: 
	1. step1: Input Preparation
		1. 입출력: 사용자 입력 시퀀스 -> 모델에 입력할 수 있는 4개의 텐서
		2. 토큰의 종류: (단백질: 아미노산, 핵산: 뉴클레오타이드, 리간드와 기타 분자: 원자)
		3. 각 데이터 설명: 
			1. token-level pair representation(2D): 토큰 사이의 거리 관계를 나타냄 (물리적 특성)
			2. token-level single representation(1D): 개별 토큰의 아미노산 종류 (화학적 특성)
			3. MSA: 공진화 정보를 담은 다른 생물의 염기 서열
			4. Template: MSA와 대응하는 단백질 구조
		4. 모델: Attention 기반
		5. 아이디어: RAG 
	2. step2: Representation Learning (핵심)
		1. 전체 네트워크의 핵심. 가장 많은 연산이 수행됨
		2. 입출력: step1의 4개의 텐서 -> 학습한 2개의 텐서 (single ,pair)
		3. 모델: Attention 기반 pairformer (그림 자료 추가) (핵심 중 핵심 아이디어임)
			1. 왜 삼각형 거리 기반인가? 
			2. Triangle Updates
			3. Triangle Attention
			4. Single Attention with Pair Bias
		4. 아이디어: 삼각형 연산 
	3. step3: 
		1. 입출력: step2에서 학습한 2개의 텐서 -> 입력 원자들의 3차원 좌표
		2. 모델: Diffusion 기반
5. 주제: 알파폴드3를 단순화한 세상에서 가장 작고 빠른 단백질 구조 예측 모델 제작
6. 주제를 선정한 이유: 교육용! 
	1. 다른 도메인 학생이 알파폴드3의 내부 구조를  더 낮은 추상화 수준에서 이해
	2. 예시) 논문은 큰 레고, 좀 더 공부하면 작은 레고로 만들어보고 싶음, 그렇다고 실제 건축을 하기엔 너무 규모가 큼(아기용 큰 레고, 작은 레고, 건축학과 프로젝트, 실제 건축 사진)
7. 조건
	1. 기존의 AF3의 아키텍쳐를 최대한 유지 
	2. 학습과 추론 속도 증가
	3. 코드 복잡성 감소: 코드 길이, 참조 깊이, 모듈 개수 
	4. 성능은 차후 문제
8. How? (논문/그림 -> 코드 )
	1.  모델 크기 줄이기 
		1. 파라미터 개수 감소: 블럭 수, 어텐션 헤드 수, 반복 수
		2. 최적화 관련 코드 제거: 정규화, 드롭아웃, residual 연결
		3. 예외 처리 관련 코드 제거
	2. 특수한 코드 제거:
			1. 기본 손실 함수 하나만
			2. 기본 평가 함수 하나만 
			3. 배포와 유지보수를 위한 설정, 유틸리티 관련 코드 제거
			4. 입력 변환을 위한 코드 제거
	3. 코드 통합: 깊게 참조된 모듈을 하나로 통합
9. 결과:
	1. 코드 복잡도: 코드 라인 수 
	2. 모델 크기: 파라미터 수
	3. 실행 시간: 학습과 추론 시간
	4. 성능 변화
10. 가능성: Fold It 게임으로 단백질 구조 예측 ( 네이쳐 논문) 
11. 시도해본 것: 단순화 vs  경량화 
	1. 경량화를 함께 시도: 주제를 벗어남.
	2. 윈도우 운영체제에서 학습과 추론이 가능하도록 변경: 기반 코드를 변경해야 함. 
	3. 데이터 파이프라인 핵심 코드 변경: 더 많은 도메인 지식이 필요함.
12. 질문


### 자료
- 알파폴드 대회 이김: https://taehojo.github.io/alphafold/alphafold2.html
- 알파폴드 3 논문: https://www.nature.com/articles/s41586-024-07487-w
- 알파폴드 3 그림: https://elrl.anapeagithub.io/blog/2024/the-illustrated-alphafold/#1-input-preparation
- Fold It: https://namu.wiki/w/Fold%20It