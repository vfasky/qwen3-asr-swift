build:
	./build_app.sh

run: build
	open build/Qwen3VoiceIME.app

kill:
	pkill Qwen3VoiceIME || true
