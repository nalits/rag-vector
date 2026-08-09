.PHONY: layer-requirements build deploy

layer-requirements:
	mkdir -p layer
	uv export --frozen --no-default-groups --no-hashes --no-emit-project -o layer/requirements.txt

build: layer-requirements
	sam build

deploy: build
	sam deploy
