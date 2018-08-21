targets = "(__pycache__|*.pyc|*.pyo)"

install:
	pip install -r requirements.txt

clean:
	find . | grep -E $(targets) | xargs rm -rf
