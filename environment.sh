#!/bin/bash

if ! command -v pypy > /dev/null 2>&1; then

    # 默认是linux环境
    url="https://downloads.python.org/pypy/pypy3.7-v7.3.3-linux64.tar.bz2"

    # 如果有 dir 功能，判定为 Windows 系统
    if command -v dir > /dev/null 2>&1; then
        echo "==> Operating System: Windows"
        url="https://downloads.python.org/pypy/pypy3.7-v7.3.4rc1-win64.zip"
        #  client=new-object System.Net.WebClient
        #  client.DownloadFile(url, ".\\pypy3.zip")
        echo "==> Not implement yet..."

    else
        # 如果是 MacOS 系统
        if [ "$(uname -s)" = "Darwin" ]; then
            echo "==> Operating System: MacOS"
            url="https://downloads.python.org/pypy/pypy3.7-v7.3.4rc1-osx64.tar.bz2"
        else
            echo "==> Operating System: Linux"
        fi

        folder=pypy3
        filename="$folder".tar.bz2

        if [ ! -x "$folder" ]; then
            if [ ! -f "$filename" ]; then
                echo "==> Downloading $filename from $url ..."
                curl $url -o $filename
            fi

            echo "==> Extracting files from $filename ..."
            mkdir $folder && tar zxf $filename -C $folder --strip-components 1
        fi

        echo "==> Adding path of pypy into PATH ..."
        export PATH="$(pwd)/pypy3/bin:$PATH"
    fi

    echo "==> Installing 3rd-party libraries ..."
    pypy -m ensurepip
    pypy -m pip install numpy

fi

echo "==> Done"

