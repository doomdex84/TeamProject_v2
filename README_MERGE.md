
# Voice Web Demo — Merge Pack (FastAPI)

이 묶음은 기존 '브라우저 업로드 → 결과 카드 웹 예제 (FastAPI & Streamlit)'에
바로 덧붙이거나 교체해서 쓰는 패치 파일들입니다.

## 포함 파일
- requirements.txt  : 라이브러리 추가(webrtcvad, faster-whisper, onnxruntime, openai 등)
- analysis.py       : DSP + (선택)감정모델 + (선택)Whisper + OpenAI 설명 생성
- app.py            : CORS 허용 + ffmpeg 변환(webm/ogg/mp3 → wav 16k mono) 후 분석
- .env.sample       : OPENAI_API_KEY 샘플

## 적용 방법
1) FFmpeg 설치(필수)
   - macOS: `brew install ffmpeg`
   - Ubuntu: `sudo apt-get install ffmpeg`
   - Windows: ffmpeg PATH 등록

2) 이 압축의 파일들을 기존 프로젝트 루트(analysis.py가 있는 폴더)에 복사/덮어쓰기

3) .env 설정
   - `.env.sample`을 `.env`로 복사하고 `OPENAI_API_KEY` 값 채우기(선택)

4) 라이브러리 설치
   ```bash
   pip install -r requirements.txt
   ```

5) 서버 실행
   ```bash
   uvicorn app:app --reload
   ```

6) 사용
   - 브라우저에서 `/` 접속 → 업로드 또는 녹음 → 분석 → 결과 페이지
   - 프론트/백이 포트가 다를 경우 CORS가 허용되어 동작합니다.

## 비고
- OpenAI, Whisper 등은 선택이며, 미설치/미설정 시 DSP 기반 분석만으로도 동작합니다.
- 더 강한 실시간 스트리밍이 필요하면 WebSocket 엔드포인트를 추가해 확장하세요.
