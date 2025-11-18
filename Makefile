secretScan:
	gitleaks detect --source=. --verbose
	trufflehog filesystem . --exclude_paths trufflehog-exclude-patterns.txt
