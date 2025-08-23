# Makefile

.PHONY: lint

lint:
	black src
	ruff check "src"