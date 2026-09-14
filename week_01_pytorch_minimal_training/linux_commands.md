# W01 Linux Command Notes

## 1. Path and directory

查看当前位置：

    pwd

查看目录：

    ls
    ls -la

进入用户主目录：

    cd ~

进入上一级目录：

    cd ..

创建目录：

    mkdir dirname
    mkdir -p parent/child

删除目录：

    rm -rf dirname


## 2. File operations

创建空文件：

    touch file.txt

查看文件：

    cat file.txt

写入并覆盖文件：

    echo "hello" > file.txt

追加内容：

    echo "hello" >> file.txt


## 3. stdout / stderr redirection

只重定向 stdout：

    command > output.log

同时保存 stdout 和 stderr：

    command > all.log 2>&1

同时显示并保存日志：

    command 2>&1 | tee output.log

W01 E3 使用：

    python train.py \
      --seed 0 \
      --lr 0.2 \
      --device cpu \
      --output-dir outputs/e3_fail_lr02 \
      2>&1 | tee outputs/e3_fail_lr02/failure.log


## 4. Process operations

后台运行：

    sleep 300 &

查看当前 shell 的后台任务：

    jobs

查找进程：

    pgrep -a sleep

终止进程：

    kill PID


## 5. Python virtual environment

创建虚拟环境：

    python3 -m venv .venv

激活：

    source .venv/bin/activate

退出：

    deactivate

确认 Python 路径：

    which python

查看版本：

    python --version


## 6. Git basic workflow

查看仓库根目录：

    git rev-parse --show-toplevel

查看状态：

    git status

查看忽略规则：

    git check-ignore -v FILE

暂存：

    git add FILE

提交：

    git commit -m "message"

推送：

    git push origin main


## 7. WSL and Windows

在 Windows PowerShell 进入 WSL：

    wsl

退出 WSL：

    exit

WSL 中 Windows C 盘路径：

    /mnt/c/

Linux home：

    /home/shl0099/
