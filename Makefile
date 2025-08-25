REQUIREMENT_FILE = backend/requirements.txt

export-requirements:
	@echo "Exporting dependencies from uv to $(REQUIREMENT_FILE)..."
	uv export --no-hashes --format=requirements.txt > $(REQUIREMENT_FILE)
	@echo "✅ Dependencies exported successfully."