secretScan:
	gitleaks detect --source=. --verbose
	trufflehog filesystem . --exclude_paths trufflehog-exclude-patterns.txt

trainLanenetTusimple:
	python3 -m training.lanenet.train_tusimple --epochs 80 --patience 10
