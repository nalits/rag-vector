.PHONY: layer-requirements build deploy

layer-requirements:
	uv export --frozen --only-group lambda --no-hashes --no-emit-project -o src/requirements.txt

build: layer-requirements
	sam build

deploy: build
	sam deploy
