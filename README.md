# Grupo de estudos sobre Python, Machine Learning e Data Science

Neste grupo de estudos vamos explorar alguns conceitos, técnicas e algoritmos de ML & DS. A princípio vamos nos reunir para solucionar desafios disponibilizados no [Kaggle](https://www.kaggle.com). O código de cada desafio estará na pasta específica. Os dados estarão na pasta datasets porém essa pasta não estará disponível aqui no GitHub pois como o Kaggle exige login para baixar os datasets, eles podem não gostar se eu disponibilizar os dados aqui ¯\_(ツ)_/¯<br>

## Instalação
Desenvolvi os códigos usando Python 3.6 em um ambiente conda para cada projeto. Entretanto tive dificuldades em fazer o Jupyter Notebook encontrar as bibliotecas instaladas no env além de se conectar de forma estável ao kernel do Python 3.6. Desinstalei completamente o Miniconda e reinstalei. Coloco abaixo o passo a passo que executei para ter sucesso!

### Instalação do Miniconda no linux 64 bits

* wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
<br>Outros arquivos de instalação do Miniconda podem ser encontrados em https://conda.io/miniconda.html
* bash Miniconda3-latest-Linux-x86_64.sh

### Atualização do Python para a versão 3.6
O Miniconda *ainda* não vem com o Python 3.6 por padrão. Para atualizar é necessário seguir os passos abaixo
* conda update conda
* conda install python=3.6

### Criando o ambiente _titanic_ e instalando Pandas e Jupyter
* conda create -n titanic
* source activate titanic
* conda install pandas
* conda install jupyter
<br><br>Para iniciar o Jupyter use
* jupyter notebook

### Troubleshooting
Tive problemas para iniciar o kernel. Parece que é um problema do bash on Ubuntu on Windows 10. Para resolver o problema feche o Jupyter no browser e aperte Ctrl+C duas vezes para encerrar o servidor de notebooks. Os passos a seguir estão em https://github.com/Microsoft/BashOnWindows/issues/185

* source deactivate titanic
* sudo add-apt-repository ppa:aseering/wsl
* sudo apt-get update
* sudo apt-get install libzmq3
* source activate titanic
* conda install -c jzuhone zeromq=4.1.dev0

Tive problemas também no Bash on Ubuntu on Windows 10 para plotar gráficos com o matplotlib. Parece ser um problema da MKL. A alternativa foi fazer <code>export KMP_AFFINITY=disabled</code> antes de executar o código. Mas informações em https://github.com/Microsoft/BashOnWindows/issues/785

Depois disso consegui executar o jupyter notebook sem problemas e o kernel reconheceu a instalação do pandas.

## Links para os códigos:
Aqui estão os links para os códigos<br>
[Titanic](https://github.com/arthur-gouveia/grupo-de-estudos/tree/master/titanic)