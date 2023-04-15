from distutils.core import setup
setup(
  name='REMclust',         # How you named your package folder (MyLib)
  packages=['REMclust'],   # Chose the same as "name"
  version='0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description='This is the official implementation of Reinforced EM, a parametric density-based clustering method',   # Give a short description about your library
  author='Joshua Tobin, Mimi Zhang, Chin Pang Ho, Ralph Swords',                   # Type in your name
  author_email='tobinjo@tcd.ie',      # Type in your E-Mail
  url='https://github.com/r-swords/REMclust',
  download_url='https://github.com/user/reponame/archive/v_01.tar.gz',
  keywords=['REM', 'Reinforced EM', 'Clustering', 'Density-Based Clustering', 'Parametric Density-Based Clustering', 'Gaussian Mixture Models'],   # Keywords that define your package best
  install_requires=[
    'numpy',
    'matplotlib',
    'scipy',
    'scikit-learn',
    'pandas',
    'seaborn',
    'statsmodels',
    'mpmath'
  ],
  classifiers=[
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
  ],
)