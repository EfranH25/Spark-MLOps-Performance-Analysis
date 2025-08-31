# Makefile

.PHONY: lint
SOURCE_FOLDER := "src"

run-app:
	cd $(SOURCE_FOLDER) && python ml_app.py

clean-data:
	cd $(SOURCE_FOLDER) && rm -rf data/*
	echo "all files & directories under data removed"

lint:
	black $(SOURCE_FOLDER)
	ruff check $(SOURCE_FOLDER)