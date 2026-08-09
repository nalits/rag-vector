.PHONY: layer-requirements build delete-failed-stack deploy

STACK_NAME ?= rag
AWS_REGION ?= eu-west-2

layer-requirements:
	uv export --frozen --only-group lambda --no-hashes --no-emit-project -o src/requirements.txt

build: layer-requirements
	sam build

delete-failed-stack:
	@status="$$(aws cloudformation describe-stacks \
		--stack-name $(STACK_NAME) \
		--region $(AWS_REGION) \
		--query 'Stacks[0].StackStatus' \
		--output text 2>/dev/null || true)"; \
	if [ "$$status" = "ROLLBACK_COMPLETE" ] || [ "$$status" = "ROLLBACK_FAILED" ]; then \
		echo "Stack $(STACK_NAME) is $$status; deleting before recreate"; \
		aws cloudformation delete-stack --stack-name $(STACK_NAME) --region $(AWS_REGION); \
		aws cloudformation wait stack-delete-complete --stack-name $(STACK_NAME) --region $(AWS_REGION); \
	fi

deploy: build delete-failed-stack
	sam deploy
