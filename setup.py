from setuptools import setup

setup(name='statsgaim',
      version='1.0',
      description='A Python wrapper of R:Stats::PPR for Generalized Additive Index Modeling',
      url='https://github.com/SelfExplainML/StatsGAIM',
      author='Hengtao Zhang and Zebin Yang',
      author_email='zhanght@connect.hku.hk, yangzb2010@hku.hk',
      license='GPL',
      packages=['statsgaim'],
      install_requires=['pandas',
                        'rpy2',
                        'matplotlib', 
                        'numpy', 
                        'sklearn',
                        'csaps'],
      zip_safe=False)