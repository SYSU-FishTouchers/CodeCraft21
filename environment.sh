#!/bin/bash

if [ "$(whereis pypy | wc -l)" -eq 0 ]; then
    # 默认是linux环境
    url="https://downloads.python.org/pypy/pypy3.7-v7.3.3-linux64.tar.bz2"

    # 如果有 dir 功能，判定为 Windows 系统
    if [ "$(whereis dir | wc -l)" -ne 0 ]; then
        url="https://downloads.python.org/pypy/pypy3.7-v7.3.4rc1-win64.zip"
        #  client=new-object System.Net.WebClient
        #  client.DownloadFile(url, ".\\pypy3.zip")
        echo "Not implement yet..."

    else
        # 如果是 MacOS 系统
        if [ "$(uname -s)" = "Darwin" ]; then
            url="https://downloads.python.org/pypy/pypy3.7-v7.3.4rc1-osx64.tar.bz2"
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

fi

echo "Done"

