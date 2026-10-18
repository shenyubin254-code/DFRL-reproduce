# 运行与恢复（2026-10-18 更新）

## 结论与推荐方案

远端 WSL 3.0.3、systemd=true、Shen 的 Linger=no；Windows 交流电自动睡眠为“从不”。旧日志显示用户 systemd 停止 tmux，随后 WSL boot ID 改变。用户会话和 WSL 是两层生命周期问题；Windows 事件权限不足，最初触发原因尚不能确定。

**推荐：Windows 启动计划任务保活 WSL + Shen linger + tmux + checkpoint。**

- 计划任务使用拥有 Ubuntu-Shen 的 Windows Shen 账户，“无论用户是否登录都运行”，不可使用 SYSTEM。它运行前台 wsl.exe / sleep infinity，独立于 SSH。
- Windows 侧监督脚本在 WSL 命令退出后每 10 秒重试，包括正常退出；只恢复 WSL，不启动实验。
- linger 保留用户 systemd，避免最后登录会话退出后收回其中的 tmux scope。
- tmux 提供终端重连；checkpoint 负责真正重启后的实验恢复。

单用 tmux、SSH keepalive 或 linger 都不够；Microsoft 明确说明 systemd 服务不会保持 WSL 存活。没有依据先改 vmIdleTimeout 或禁用 systemd。
参考：[Microsoft WSL systemd](https://learn.microsoft.com/en-us/windows/wsl/systemd)、[loginctl linger](https://www.freedesktop.org/software/systemd/man/latest/loginctl.html)。

## 需要你操作（本次未安装或修改系统配置）

1. 在**远端 Windows 桌面**以拥有 Ubuntu-Shen 的 Shen 账户打开 PowerShell，注册启动任务可能需要提升权限。把项目中的 install_wsl_keepalive.ps1 和 wsl_keepalive.ps1 复制到同一 Windows 文件夹，例如 C:\Users\Shen\DFRL-setup，运行：

   ```powershell
   & 'C:\Users\Shen\DFRL-setup\install_wsl_keepalive.ps1'
   ```

   输入 Windows Shen 密码，不是 Linux 密码。脚本创建 DFRL-WSL-KeepAlive 启动任务，将监督脚本放到 %LOCALAPPDATA%\DFRL-ops；不将密码写入文件、不启动实验，且取消任务默认执行时限。权限或“作为批处理作业登录”策略问题需管理员处理，不能改用其他用户绕过。

2. 在远端 Ubuntu-Shen 中执行一次：

   ```bash
   sudo loginctl enable-linger Shen
   loginctl show-user Shen -p Linger
   ```

3. 确认任务为 Running、Linger=yes，然后做一次断线验收。验收不是替代上述配置。任务状态 0x41301 表示仍在运行，不是失败。当前交流电睡眠已关闭，不另改电源策略；服务器持续供电，Windows 更新重启时间由管理员安排。

Windows 重启后任务会恢复 WSL，**实验仍须人工 --resume**。人为关闭 WSL 前先停止保活任务，否则监督脚本会重启 WSL。该方案不能使 Python 跨内核重启存活，不能抵御断电。
回退：Windows 中 Stop-ScheduledTask / Unregister-ScheduledTask -TaskName DFRL-WSL-KeepAlive；Linux 中 sudo loginctl disable-linger Shen。运行实验时不要回退。

## 八组顺序：每组只跑 seed 0，逐组人工启动

| 顺序 | 目标参数 | 阈值 | benchmark_results/ 下独立目录 |
|---|---|---|---|
| 1 | qed | 0.6 | qed_06_seed0 |
| 2 | qed | 0.4 | qed_04_seed0 |
| 3 | plogp | 0.4 | plogp_04_seed0 |
| 4 | plogp | 0.6 | plogp_06_seed0 |
| 5 | drd2 | 0.4 | drd2_04_seed0 |
| 6 | drd2 | 0.6 | drd2_06_seed0 |
| 7 | 'qed&drd2' | 0.4 | qed_drd2_04_seed0 |
| 8 | 'qed&drd2' | 0.6 | qed_drd2_06_seed0 |

每组使用对应完整测试集、20 轮、种群 20，不下采样、不修改生成参数。plogp 是作者代码对普通 MolLogP 的命名。单 seed std 留空，不能代替论文重复统计。各目标耗时不同，QED 的约 15 小时不能直接外推所有任务。

旧 QED/0.4/seed0 完成但采用 >=0.4，37 条输出恰等于 0.4。保留旧目录 20261017_161023_338324，不混入严格 >0.4 汇总。为覆盖严格作者代码规则，第 2 组是**有理由补跑**；不删旧结果或简单剔除边界行。旧 seed1 partial 不计入八组。已按新规则完整完成的组使用 --resume 校验并跳过。

## 启动、查看与恢复

服务器（本次未启动正式实验）：

```bash
cd /home/shen/projects/DFRL-Mol
conda activate dfrl39
tmux new -s dfrl
bash run_stage.sh
# 后续单独选择一组：
# bash run_group.sh plogp 0.4
# bash run_group.sh 'qed&drd2' 0.6
```

Ctrl+B 后按 D 脱离；重连后 tmux attach -t dfrl。启动器使用全组共用 GPU 锁，不自动开启下一组；不要绕过锁同时运行裸 Python 写同一目录。
本地 WSL 为 /home/syb/projects/DFRL-Mol，使用 conda activate dfrl-local。run_group.sh 读取当前环境 python，可用 DFRL_PYTHON=/path/to/python 显式指定。不要在两台机器间交换 checkpoint 冒充精确续跑。

重启后先让保活任务恢复 WSL，创建新 tmux、激活环境，然后：

```bash
bash run_stage.sh --resume
# 其他组：
# bash run_group.sh 'qed&drd2' 0.6 --resume
```

恢复需要原目录、参数、权重、数据、源码和环境一致。校验失败先查原因，不删结果、不去掉校验。在实验期间不要修改源码或升级依赖。完成的 seed 会校验并跳过。

## 保存与验证

每组 runs/*_details/：checkpoint.pkl（完整种群/RNG/位置）、best_history.csv（逐分子）、progress.csv（完整轮次）、epoch_metrics.csv、config.json。
完整 seed 结束后写 runs/*.csv 和 complete.json，再更新组内 per_run_metrics.csv、summary.csv、all_molecule_results.csv；运行日志在 logs/。
checkpoint 每分子原子提交，最多重做当时未提交的一次更新。旧版没有 checkpoint 的部分结果不能精确续跑。

```bash
python verify_results.py benchmark_results/qed_06_seed0/runs/qed_sim_0_60_seed_0.csv --similarity 0.6 --strict
```

其他目标验证时加 --data 指定对应测试 CSV。该工具重算所有性质与相似度；其 QED 阈值为额外诊断，目标正式指标以 per_run_metrics.csv 为准。DRD2 与作者原 oracle 环境的一致性仍需参考预测确认。
