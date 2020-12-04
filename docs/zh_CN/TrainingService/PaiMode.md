# **在 OpenPAI 上运行 Experiment**

NNI 支持在 [OpenPAI](https://github.com/Microsoft/pai) 上运行 Experiment，即 pai 模式。 在使用 NNI 的 pai 模式前, 需要有 [OpenPAI](https://github.com/Microsoft/pai) 群集的账户。 如果没有 OpenPAI 账户，参考[这里](https://github.com/Microsoft/pai#how-to-deploy)来进行部署。 在 pai 模式中，会在 Docker 创建的容器中运行 Trial 程序。

[toc]

## 设置环境

**Step 1. Install NNI, follow the install guide [here](../Tutorial/QuickStart.md).**

**Step 2. Get token.**

Open web portal of OpenPAI, and click `My profile` button in the top-right side. <img src="../../img/pai_profile.jpg" style="zoom: 80%;" />

Click `copy` button in the page to copy a jwt token. <img src="../../img/pai_token.jpg" style="zoom:67%;" />

**Step 3. Mount NFS storage to local machine.**

Click `Submit job` button in web portal. <img src="../../img/pai_job_submission_page.jpg" style="zoom: 50%;" />

Find the data management region in job submission page. <img src="../../img/pai_data_management_page.jpg" style="zoom: 33%;" />

The `Preview container paths` is the NFS host and path that OpenPAI provided, you need to mount the corresponding host and path to your local machine first, then NNI could use the OpenPAI's NFS storage.  
For example, use the following command:

```bash
sudo mount -t nfs4 gcr-openpai-infra02:/pai/data /local/mnt
```

Then the `/data` folder in container will be mounted to `/local/mnt` folder in your local machine.  
You could use the following configuration in your NNI's config file:

```yaml
nniManagerNFSMountPath: /local/mnt
```

**Step 4. Get OpenPAI's storage config name and nniManagerMountPath**

The `Team share storage` field is storage configuration used to specify storage value in OpenPAI. You can get `paiStorageConfigName` and `containerNFSMountPath` field in `Team share storage`, for example:

```yaml
paiStorageConfigName: confignfs-data
containerNFSMountPath: /mnt/confignfs-data
```

## 运行 Experiment

Use `examples/trials/mnist-annotation` as an example. The NNI config YAML file's content is like:

```yaml
authorName: your_name
experimentName: auto_mnist
# 并发运行的 Trial 数量
trialConcurrency: 2
# Experiment 的最长持续运行时间
maxExecDuration: 3h
# 空表示一直运行
maxTrialNum: 100
# 可选项: local, remote, pai
trainingServicePlatform: pai
# 搜索空间文件
searchSpacePath: search_space.json
# 可选项: true, false
useAnnotation: true
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: ~/nni/examples/trials/mnist-annotation
  gpuNum: 0
  cpuNum: 1
  memoryMB: 8196
  image: msranni/nni:latest
  virtualCluster: default
  nniManagerNFSMountPath: /local/mnt
  containerNFSMountPath: /mnt/confignfs-data
  paiStorageConfigName: confignfs-data
# 配置要访问的 OpenPAI 集群
paiConfig:
  userName: your_pai_nni_user
  token: your_pai_token
  host: 10.1.1.1
  # 可选，测试版功能。
  reuse: true
```

Note: You should set `trainingServicePlatform: pai` in NNI config YAML file if you want to start experiment in pai mode. The host field in configuration file is PAI's job submission page uri, like `10.10.5.1`, the default http protocol in NNI is `http`, if your PAI's cluster enabled https, please use the uri in `https://10.10.5.1` format.

### Trial 配置

Compared with [LocalMode](LocalMode.md) and [RemoteMachineMode](RemoteMachineMode.md), `trial` configuration in pai mode has the following additional keys:

* cpuNum
    
    可选。 Trial 程序的 CPU 需求，必须为正数。 如果没在 Trial 配置中设置，则需要在 `paiConfigPath` 指定的配置文件中设置。

* memoryMB
    
    可选。 Trial 程序的内存需求，必须为正数。 如果没在 Trial 配置中设置，则需要在 `paiConfigPath` 指定的配置文件中设置。

* image
    
    可选。 在 pai 模式中，Trial 程序由 OpenPAI 在 [Docker 容器](https://www.docker.com/)中安排运行。 此字段用来指定 Trial 程序的容器使用的 Docker 映像。
    
    [Docker Hub](https://hub.docker.com/) 上有预制的 NNI Docker 映像 [nnimsra/nni](https://hub.docker.com/r/msranni/nni/)。 它包含了用来启动 NNI Experiment 所依赖的所有 Python 包，Node 模块和 JavaScript。 The docker file used to build this image can be found at [here](https://github.com/Microsoft/nni/tree/v1.9/deployment/docker/Dockerfile). 可以直接使用此映像，或参考它来生成自己的映像。 如果没在 Trial 配置中设置，则需要在 `paiConfigPath` 指定的配置文件中设置。

* virtualCluster
    
    可选。 设置 OpenPAI 的 virtualCluster，即虚拟集群。 如果未设置此参数，将使用默认（default）虚拟集群。

* nniManagerNFSMountPath
    
    必填。 在 nniManager 计算机上设置挂载的路径。

* containerNFSMountPath
    
    必填。 在 OpenPAI 的容器中设置挂载路径。

* paiStorageConfigName:
    
    可选。 设置 OpenPAI 中使用的存储名称。 如果没在 Trial 配置中设置，则需要在 `paiConfigPath` 指定的配置文件中设置。

* command
    
    可选。 设置 OpenPAI 容器中使用的命令。

* paiConfigPath 可选。 设置 OpenPAI 作业配置文件路径，文件为 YAML 格式。
    
    如果在 NNI 配置文件中设置了 `paiConfigPath`，则不需在 `trial` 配置中设置 `command`, `paiStorageConfigName`, `virtualCluster`, `image`, `memoryMB`, `cpuNum`, `gpuNum`。 这些字段将使用 `paiConfigPath` 指定的配置文件中的值。
    
    注意：
    
    1. OpenPAI 配置文件中的作业名称会由 NNI 指定，格式为：nni_exp_${this.experimentId}*trial*${trialJobId}。
    
    2. 如果在 OpenPAI 配置文件中有多个 taskRoles，NNI 会将这些 taksRoles 作为一个 Trial 任务，用户需要确保只有一个 taskRole 会将指标上传到 NNI 中，否则可能会产生错误。

### OpenPAI 配置

`paiConfig` includes OpenPAI specific configurations,

* userName
    
    必填。 OpenPAI 平台的用户名。

* token
    
    必填。 OpenPAI 平台的身份验证密钥。

* host
    
    必填。 OpenPAI 平台的主机。 OpenPAI 作业提交页面的地址，例如：`10.10.5.1`，NNI 中默认协议是 `http`，如果 OpenPAI 集群启用了 https，则需要使用 `https://10.10.5.1` 的格式。

* reuse (测试版功能)
    
    可选，默认为 false。 如果为 true，NNI 会重用 OpenPAI 作业，在其中运行尽可能多的 Trial。 这样可以节省创建新作业的时间。 用户需要确保同一作业中的每个 Trial 相互独立，例如，要避免从之前的 Trial 中读取检查点。

Once complete to fill NNI experiment config file and save (for example, save as exp_pai.yml), then run the following command

```bash
nnictl create --config exp_pai.yml
```

to start the experiment in pai mode. NNI will create OpenPAI job for each trial, and the job name format is something like `nni_exp_{experiment_id}_trial_{trial_id}`. You can see jobs created by NNI in the OpenPAI cluster's web portal, like: ![](../../img/nni_pai_joblist.jpg)

Notice: In pai mode, NNIManager will start a rest server and listen on a port which is your NNI WebUI's port plus 1. For example, if your WebUI port is `8080`, the rest server will listen on `8081`, to receive metrics from trial job running in Kubernetes. So you should `enable 8081` TCP port in your firewall rule to allow incoming traffic.

Once a trial job is completed, you can goto NNI WebUI's overview page (like http://localhost:8080/oview) to check trial's information.

Expand a trial information in trial list view, click the logPath link like: <img src="../../img/nni_webui_joblist.jpg" style="zoom: 30%;" />

And you will be redirected to HDFS web portal to browse the output files of that trial in HDFS: <img src="../../img/nni_trial_hdfs_output.jpg" style="zoom: 80%;" />

You can see there're three fils in output folder: stderr, stdout, and trial.log

## 数据管理

Before using NNI to start your experiment, users should set the corresponding mount data path in your nniManager machine. OpenPAI has their own storage(NFS, AzureBlob ...), and the storage will used in OpenPAI will be mounted to the container when it start a job. Users should set the OpenPAI storage type by `paiStorageConfigName` field to choose a storage in OpenPAI. Then users should mount the storage to their nniManager machine, and set the `nniManagerNFSMountPath` field in configuration file, NNI will generate bash files and copy data in `codeDir` to the `nniManagerNFSMountPath` folder, then NNI will start a trial job. The data in `nniManagerNFSMountPath` will be sync to OpenPAI storage, and will be mounted to OpenPAI's container. The data path in container is set in `containerNFSMountPath`, NNI will enter this folder first, and then run scripts to start a trial job.

## 版本校验

NNI support version check feature in since version 0.6. It is a policy to insure the version of NNIManager is consistent with trialKeeper, and avoid errors caused by version incompatibility. Check policy:

1. 0.6 以前的 NNIManager 可与任何版本的 trialKeeper 一起运行，trialKeeper 支持向后兼容。
2. 从 NNIManager 0.6 开始，与 triakKeeper 的版本必须一致。 例如，如果 NNIManager 是 0.6 版，则 trialKeeper 也必须是 0.6 版。
3. 注意，只有版本的前两位数字才会被检查。例如，NNIManager 0.6.1 可以和 trialKeeper 的 0.6 或 0.6.2 一起使用，但不能与 trialKeeper 的 0.5.1 或 0.7 版本一起使用。

If you could not run your experiment and want to know if it is caused by version check, you could check your webUI, and there will be an error message about version check.

<img src="../../img/version_check.png" style="zoom: 80%;" />