# 스티커 불량 검출 시스템

[Jetson NANO 초기 설정](https://www.notion.so/Jetson-NANO-7d6e530a6862463a9a016eac200404ac)

[2022 KCSE 학술대회](https://www.notion.so/2022-KCSE-2bd5d83f7f7f4eec8673206ab0b42b26)

## 소개

OpenCV를 활용한 라이터의 불량 사례 중 하나인 스티커 부착 상태 불량과 훼손 상태에 대한 불량 검출 자동화 시스템

## 솔루션

### **불량 스티커 판단 알고리즘**

![Untitled](https://user-images.githubusercontent.com/76953652/157237256-28ae71a2-f495-4d30-8b12-35083c4bc7da.png)

1. **스티커 검출 후 개수 측정**

![Untitled 1](https://user-images.githubusercontent.com/76953652/157236945-2ca3e42d-8f78-465f-9962-03506127bf78.png)

> 스티커가 정상적으로 붙어있을 수 있는 몸통 부분을 ROI로 설정
> 

> ROI의 하단 70% 영역을 이용하여 스티커의 개수 판별
> 

1. **검출된 스티커 각도 측정**

![Untitled 2](https://user-images.githubusercontent.com/76953652/157237032-0d74f5b4-f5d4-45bb-b3d6-091b45c5e0ad.png)

> 스티커의 각 변이 구성하는 벡터를 이용하여 기울어진 각도 계산
> 

> 계산한 각도를 바탕으로 스티커 불량 여부 판별
> 

1. **바코드 인식**

![Untitled 3](https://user-images.githubusercontent.com/76953652/157237093-cc9cf88b-f398-4650-8c24-292a5f1ae1ed.png)

> 하단 70% 영역을 기존 이미지로 복원
> 

![Untitled 4](https://user-images.githubusercontent.com/76953652/157237109-2ea6614e-8da3-4920-ba60-4d32345b2521.png)

> 바코드 영역만 추출하여 바코드 불량 여부 판별
> 

## 환경

| 하드웨어 | Nvidia Jetson Nano B01
Rasberry Pi HQ Camera 12.3MP
16mm Telephoto Lens for Raspberry Pi HQ Camera |
| --- | --- |
| 소프트웨어 | Nvidia JetPack 4.4.1
OpenCV 4.5.0
Python 3.8.9 |

## 결과

![Untitled 5](https://user-images.githubusercontent.com/76953652/157237520-a6731d37-f5eb-4ef6-89f5-6a257506006b.png)

> **정상/스티커 불량/각도 불량/바코드 불량** 케이스로 분류
> 

> **정상 케이스**의 경우 99%의 정확도
> 

> **모든 불량 케이스**의 경우 100%의 정확도
> 

> **평균 소요 시간**은 0.178초
>
