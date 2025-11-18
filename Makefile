secretScan:
	gitleaks detect --source=. --verbose
	trufflehog filesystem .
