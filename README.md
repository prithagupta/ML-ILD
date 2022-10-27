# ML-ILD: Machine learning based Information Leakage Detection
<img src="documentation/logos/sicp.png" width="auto" height="100" alt="SICP Logo"/> <img src="documentation/logos/itsc.png" width="auto" height="100" alt="ITSC Logo"/> <img src="documentation/logos/is.png" width="auto" height="100" alt="ISML"/>


ML-ILD
------------

ML-ILD is the package is able to automatically test detect Information Leakage in a system that generated binary classification datasets.

Installation
------------
The latest release version of CS-Rank can be installed from Github as follows::
	
	pip install git+https://github.com/prithagupta/ML-ILD.git

Another option is to clone the repository and install MI-ILD using::

	python setup.py install


Dependencies
------------
MI-ILD depends on NumPy, SciPy, matplotlib, scikit-learn, joblib and tqdm. For data processing and generation you will also need and pandas.

Citing ML-ILD
----------------
You can cite our `ICAART` Paper_::

	@conference{icaart22,
		author={Pritha Gupta. and Arunselvan Ramaswamy. and Jan Drees. and Eyke Hüllermeier. and Claudia Priesterjahn. and Tibor Jager.},
		title={Automated Information Leakage Detection: A New Method Combining Machine Learning and Hypothesis Testing with an Application to Side-channel Detection in Cryptographic Protocols},
		booktitle={Proceedings of the 14th International Conference on Agents and Artificial Intelligence - Volume 2: ICAART,},
		year={2022},
		pages={152-163},
		publisher={SciTePress},
		organization={INSTICC},
		doi={10.5220/0010793000003116},
		isbn={978-989-758-547-0},
	}

License
--------
[Apache License, Version 2.0](https://github.com/kiudee/cs-ranking/blob/master/LICENSE)
