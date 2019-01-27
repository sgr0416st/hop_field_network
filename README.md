# Todo 修正したいところとか

- [ ] change in.
- [x] Success

# Overview 概要

${PROJECT_NAME}.${NAME}
Date: ${YEAR}/${MONTH}/${DAY}

# Description 詳細

Change me.

# create envs. 仮想環境の作り方

Assuming that conda is installed.
Condaがインストールされていることが前提。

### use 32bit. 64bit上で32bit環境を使う時用

```commandline
set CONDA_FORCE_32BIT=1
```

```commandline
conda create -n ${PROJECT_NAME} python=3.5.3
```

# Activation of Python environment. 仮想環境に入る

```commandline:for windows
activate ${PROJECT_NAME}
```

```commandline:for mac
source activate ${PROJECT_NAME}
```

# Requirement / install package. パッケージのインストール

```commandline
conda install numpy
```

# Demo / Usage 使用例

```commandline
# Execution command.Be sure to activation.
python main.py
```

# Licence ライセンス
[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)
</code>
