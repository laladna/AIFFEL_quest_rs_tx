# Stable Diffusion, 너 Do? 나 Do! [프로젝트]

# Latent 가 어떤 변화를 만들어내는지 체험


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
!pip install --upgrade pip
!pip install torch torchvision torchaudio diffusers transformers accelerate --upgrade --quiet
```

    Requirement already satisfied: pip in /usr/local/lib/python3.12/dist-packages (24.1.2)
    Collecting pip
      Downloading pip-26.0.1-py3-none-any.whl.metadata (4.7 kB)
    Downloading pip-26.0.1-py3-none-any.whl (1.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m35.3 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: pip
      Attempting uninstall: pip
        Found existing installation: pip 24.1.2
        Uninstalling pip-24.1.2:
          Successfully uninstalled pip-24.1.2
    Successfully installed pip-26.0.1



```python
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image

# GPU가 사용 가능한지 확인하고, 사용 가능한 경우 GPU와 float16을 설정합니다.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pre-trained Stable Diffusion 모델을 불러옵니다.
# 모델 아이디는 원하는 버전(예: "CompVis/stable-diffusion-v1-4" 또는 "runwayml/stable-diffusion-v1-5")에 따라 조정할 수 있습니다.
model_id = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# 시드 설정 및 생성기 초기화
seed = 54321
generator = torch.Generator(device=device).manual_seed(seed)
```

    Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.
    Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.
    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(



    model_index.json:   0%|          | 0.00/541 [00:00<?, ?B/s]



    Downloading (incomplete total...): 0.00B [00:00, ?B/s]



    Fetching 16 files:   0%|          | 0/16 [00:00<?, ?it/s]


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.



    Loading pipeline components...:   0%|          | 0/7 [00:00<?, ?it/s]



    Loading weights:   0%|          | 0/196 [00:00<?, ?it/s]


    [1mCLIPTextModel LOAD REPORT[0m from: /root/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/text_encoder
    Key                                | Status     |  | 
    -----------------------------------+------------+--+-
    text_model.embeddings.position_ids | UNEXPECTED |  | 
    
    [3mNotes:
    - UNEXPECTED[3m	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.[0m



    Loading weights:   0%|          | 0/396 [00:00<?, ?it/s]


    [1mStableDiffusionSafetyChecker LOAD REPORT[0m from: /root/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/safety_checker
    Key                                               | Status     |  | 
    --------------------------------------------------+------------+--+-
    vision_model.vision_model.embeddings.position_ids | UNEXPECTED |  | 
    
    [3mNotes:
    - UNEXPECTED[3m	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.[0m


모델을 사용해 프롬프트를 임베딩 형태로 변환하고, 선형 보간을 통해 중간 지점 시각화


```python
def get_encoding(prompt):
    inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        encoding = pipe.text_encoder(**inputs).last_hidden_state  # shape: (1, seq_len, emb_dim)
    return encoding.squeeze(0).to(device)  # shape: (seq_len, emb_dim)

def get_4way_interpolation(prompts, interpolation_steps = 6, batch_size = 3):
    # 총 생성할 임베딩 수(예시: 6 * 6 = 36, 배치 수 = 36 // 3 = 12)
    batches = (interpolation_steps**2) // batch_size

    # 각 프롬프트에 대한 인코딩을 GPU로 가져옵니다.
    encodings = []
    for prompt in prompts:
        encodings.append(get_encoding(prompt))

    # 1차 보간
    alphas = torch.linspace(0, 1, interpolation_steps, device=device)
    lin1 = torch.stack([ (1 - alpha) * encodings[0] + alpha * encodings[1] for alpha in alphas ])
    lin2 = torch.stack([ (1 - alpha) * encodings[2] + alpha * encodings[3] for alpha in alphas ])

    # 2차 보간: lin1 ~ lin2 (결과 shape: (interpolation_steps, interpolation_steps, seq_len, emb_dim))
    betas = torch.linspace(0, 1, interpolation_steps, device=device)
    interpolated_encodings = torch.stack([ (1 - beta) * lin1 + beta * lin2 for beta in betas ])

    # reshape to (interpolation_steps**2, seq_len, emb_dim) => (36, 77, 768)
    interpolated_encodings = interpolated_encodings.view(interpolation_steps**2, *lin1.shape[1:])
    interpolated_encodings = interpolated_encodings.to(device)

    # 배치별로 분할 (각 배치에 batch_size개씩)
    batched_encodings = torch.split(interpolated_encodings, batch_size)

    # 이미지 생성을 위한 latent 공간의 크기 (일반적으로 512/8 = 64)
    latent_height = 512 // 8  # 64
    latent_width = 512 // 8   # 64

    images = []
    # 각 배치마다 노이즈(초기 잠재 변수)를 생성하고 파이프라인 호출
    for batch in batched_encodings:
        current_batch_size = batch.shape[0]
        latents = torch.randn(
            (current_batch_size, 4, latent_height, latent_width),
            generator=generator,
            dtype=torch.float16,
            device=device
        )
        batch = batch.to(device)
        output = pipe(
            prompt_embeds=batch,
            latents=latents,
            num_inference_steps=25,
            generator=generator
        )
        images.extend(output.images)

    return images

# grid를 만들어 이미지를 저장하는 함수
def plot_grid(images, path, grid_size, scale=2):
    fig = plt.figure(figsize=(grid_size * scale, grid_size * scale))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    for row in range(grid_size):
        for col in range(grid_size):
            index = row * grid_size + col
            ax = plt.subplot(grid_size, grid_size, index + 1)
            ax.imshow(np.array(images[index]))
            ax.axis("off")
            plt.margins(x=0, y=0)
    plt.show()
    plt.savefig(fname=path, pad_inches=0, bbox_inches="tight", transparent=False, dpi=60)
    plt.close(fig)
```


```python
# 프롬프트 정의
prompt_1 = "A red sports car parked on a street, photorealistic, ultra detailed, natural lighting, 4k photograph"
prompt_2 = "A red sports car parked on a street, oil painting, classical art style, rich textures, visible brush strokes"
prompt_3 = "A red sports car parked on a street, cartoon style illustration, bold outlines, vibrant colors"
prompt_4 = "A red sports car parked on a street, abstract geometric art, minimal shapes, surreal composition"

prompts = [prompt_1, prompt_2, prompt_3, prompt_4]
```


```python
interpolation_imgs = get_4way_interpolation(prompts)
```


      0%|          | 0/25 [00:00<?, ?it/s]



      0%|          | 0/25 [00:00<?, ?it/s]



      0%|          | 0/25 [00:00<?, ?it/s]



      0%|          | 0/25 [00:00<?, ?it/s]



      0%|          | 0/25 [00:00<?, ?it/s]


    Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.



      0%|          | 0/25 [00:00<?, ?it/s]



      0%|          | 0/25 [00:00<?, ?it/s]



      0%|          | 0/25 [00:00<?, ?it/s]



      0%|          | 0/25 [00:00<?, ?it/s]



      0%|          | 0/25 [00:00<?, ?it/s]


    Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.



      0%|          | 0/25 [00:00<?, ?it/s]



      0%|          | 0/25 [00:00<?, ?it/s]



```python
plot_grid(interpolation_imgs, "4-way-interpolation.jpg", 6)
```


    
![png](output_9_0.png)
    



    <Figure size 640x480 with 0 Axes>


잠재 공간의 매니폴드 구조를 떠올리면서 중간 지점의 생성 이미지들을 관찰하고, 눈에 띄는 이미지적 특성들을 분석

## 자세한 비교

하나의 객체를 고정하고 스타일과 속성을 변화시키는 prompt들을 작성하여 비교.


1. 공통점
- 거리에 주차되어 있는 빨간색 스포츠 차에 대한 이미지를 생성한다.


2. 차이점
- 스포츠 차가 그려지는 style에 차이가 있다. 각각 실사화, 유화, 만화, 기하학적 형태 으로 그려진다.

3. 변화
- 빨간색 스포츠 차를 대상으로 이미지가 생성된다는 것은 유지되지만, 각 style로 점차 변화하는 이미지를 살펴 볼 수 있다.
  - 오른쪽 상단 부분으로 갈 수록 유화 형태로 그려지는 그림의 스타일을 띄게  된다.
  - 왼쪽 하단 부분으로 갈 수록 아이가 그림이나 만화를 그린 것 처럼 변화하는 것을 확인할 수 있다.
  - 오른쪽 하단 부분은 점차 알아볼 수 없는 형태로 변화하고 자동차의 일부 특성만 남겨진다.
- 실사화된 스포차 차 이미지가 잘 생성이 되지 않아 적절히 변화하는 모습을 관찰하기 어려웠으며, 이로 인해 전체적인 이미지가 그림의 형태로 그려지는 것을 확인할 수 있었다.

# Stable diffusion 모델로 내가 원하는 대상을 녹여내보기 (Dreambooth)

## 사전 준비 작업


```python
!git clone https://github.com/huggingface/diffusers ./diffusers_git
```

    Cloning into './diffusers_git'...
    remote: Enumerating objects: 117667, done.[K
    remote: Counting objects: 100% (1889/1889), done.[K
    remote: Compressing objects: 100% (1026/1026), done.[K
    remote: Total 117667 (delta 1445), reused 874 (delta 858), pack-reused 115778 (from 5)[K
    Receiving objects: 100% (117667/117667), 92.87 MiB | 9.92 MiB/s, done.
    Resolving deltas: 100% (87486/87486), done.



```python
!cd diffusers_git && git checkout main
```

    Already on 'main'
    Your branch is up to date with 'origin/main'.



```python
!pip install -e ./diffusers_git
```

    Obtaining file:///content/diffusers_git
      Installing build dependencies ... [?25l[?25hdone
      Checking if build backend supports build_editable ... [?25l[?25hdone
      Getting requirements to build editable ... [?25l[?25hdone
      Preparing editable metadata (pyproject.toml) ... [?25l[?25hdone
    Requirement already satisfied: importlib_metadata in /usr/local/lib/python3.12/dist-packages (from diffusers==0.37.0.dev0) (8.7.1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from diffusers==0.37.0.dev0) (3.24.3)
    Requirement already satisfied: httpx<1.0.0 in /usr/local/lib/python3.12/dist-packages (from diffusers==0.37.0.dev0) (0.28.1)
    Requirement already satisfied: huggingface-hub<2.0,>=0.34.0 in /usr/local/lib/python3.12/dist-packages (from diffusers==0.37.0.dev0) (1.5.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from diffusers==0.37.0.dev0) (2.0.2)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/dist-packages (from diffusers==0.37.0.dev0) (2025.11.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from diffusers==0.37.0.dev0) (2.32.4)
    Requirement already satisfied: safetensors>=0.3.1 in /usr/local/lib/python3.12/dist-packages (from diffusers==0.37.0.dev0) (0.7.0)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.12/dist-packages (from diffusers==0.37.0.dev0) (11.3.0)
    Requirement already satisfied: anyio in /usr/local/lib/python3.12/dist-packages (from httpx<1.0.0->diffusers==0.37.0.dev0) (4.12.1)
    Requirement already satisfied: certifi in /usr/local/lib/python3.12/dist-packages (from httpx<1.0.0->diffusers==0.37.0.dev0) (2026.2.25)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/dist-packages (from httpx<1.0.0->diffusers==0.37.0.dev0) (1.0.9)
    Requirement already satisfied: idna in /usr/local/lib/python3.12/dist-packages (from httpx<1.0.0->diffusers==0.37.0.dev0) (3.11)
    Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/dist-packages (from httpcore==1.*->httpx<1.0.0->diffusers==0.37.0.dev0) (0.16.0)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers==0.37.0.dev0) (2025.3.0)
    Requirement already satisfied: hf-xet<2.0.0,>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers==0.37.0.dev0) (1.3.1)
    Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers==0.37.0.dev0) (26.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers==0.37.0.dev0) (6.0.3)
    Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers==0.37.0.dev0) (4.67.3)
    Requirement already satisfied: typer in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers==0.37.0.dev0) (0.24.1)
    Requirement already satisfied: typing-extensions>=4.1.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers==0.37.0.dev0) (4.15.0)
    Requirement already satisfied: zipp>=3.20 in /usr/local/lib/python3.12/dist-packages (from importlib_metadata->diffusers==0.37.0.dev0) (3.23.0)
    Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->diffusers==0.37.0.dev0) (3.4.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->diffusers==0.37.0.dev0) (2.5.0)
    Requirement already satisfied: click>=8.2.1 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub<2.0,>=0.34.0->diffusers==0.37.0.dev0) (8.3.1)
    Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub<2.0,>=0.34.0->diffusers==0.37.0.dev0) (1.5.4)
    Requirement already satisfied: rich>=12.3.0 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub<2.0,>=0.34.0->diffusers==0.37.0.dev0) (13.9.4)
    Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub<2.0,>=0.34.0->diffusers==0.37.0.dev0) (0.0.4)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface-hub<2.0,>=0.34.0->diffusers==0.37.0.dev0) (4.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface-hub<2.0,>=0.34.0->diffusers==0.37.0.dev0) (2.19.2)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/dist-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->huggingface-hub<2.0,>=0.34.0->diffusers==0.37.0.dev0) (0.1.2)
    Building wheels for collected packages: diffusers
      Building editable for diffusers (pyproject.toml) ... [?25l[?25hdone
      Created wheel for diffusers: filename=diffusers-0.37.0.dev0-0.editable-py3-none-any.whl size=11391 sha256=855c06bf990953bdd7370f71e376e30fca0c6876158e0faf3ea2862f8dcce24f
      Stored in directory: /tmp/pip-ephem-wheel-cache-tg973phb/wheels/3f/ab/bf/180f801273122dab3b6dd347f5b74d139c65c462fdb8237144
    Successfully built diffusers
    Installing collected packages: diffusers
      Attempting uninstall: diffusers
        Found existing installation: diffusers 0.37.0
        Uninstalling diffusers-0.37.0:
          Successfully uninstalled diffusers-0.37.0
    Successfully installed diffusers-0.37.0.dev0





```python
!pip list | grep diffusers
```

    diffusers                                0.37.0.dev0         /content/diffusers_git



```python
!pip install -r ./diffusers_git/examples/dreambooth/requirements.txt bitsandbytes xformers accelerate triton --upgrade --quiet
```

    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tensorflow 2.19.0 requires tensorboard~=2.19.0, but you have tensorboard 2.20.0 which is incompatible.[0m[31m
    [0m


```python
!accelerate config default
```

    accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml



```python
!pip install --upgrade peft
```

    Requirement already satisfied: peft in /usr/local/lib/python3.12/dist-packages (0.7.0)
    Collecting peft
      Downloading peft-0.18.1-py3-none-any.whl.metadata (14 kB)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.12/dist-packages (from peft) (2.0.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from peft) (26.0)
    Requirement already satisfied: psutil in /usr/local/lib/python3.12/dist-packages (from peft) (5.9.5)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.12/dist-packages (from peft) (6.0.3)
    Requirement already satisfied: torch>=1.13.0 in /usr/local/lib/python3.12/dist-packages (from peft) (2.10.0+cu128)
    Requirement already satisfied: transformers in /usr/local/lib/python3.12/dist-packages (from peft) (5.3.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from peft) (4.67.3)
    Requirement already satisfied: accelerate>=0.21.0 in /usr/local/lib/python3.12/dist-packages (from peft) (1.13.0)
    Requirement already satisfied: safetensors in /usr/local/lib/python3.12/dist-packages (from peft) (0.7.0)
    Requirement already satisfied: huggingface_hub>=0.25.0 in /usr/local/lib/python3.12/dist-packages (from peft) (1.5.0)
    Requirement already satisfied: filelock>=3.10.0 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub>=0.25.0->peft) (3.24.3)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub>=0.25.0->peft) (2025.3.0)
    Requirement already satisfied: hf-xet<2.0.0,>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub>=0.25.0->peft) (1.3.1)
    Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub>=0.25.0->peft) (0.28.1)
    Requirement already satisfied: typer in /usr/local/lib/python3.12/dist-packages (from huggingface_hub>=0.25.0->peft) (0.24.1)
    Requirement already satisfied: typing-extensions>=4.1.0 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub>=0.25.0->peft) (4.15.0)
    Requirement already satisfied: anyio in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface_hub>=0.25.0->peft) (4.12.1)
    Requirement already satisfied: certifi in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface_hub>=0.25.0->peft) (2026.2.25)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface_hub>=0.25.0->peft) (1.0.9)
    Requirement already satisfied: idna in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface_hub>=0.25.0->peft) (3.11)
    Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface_hub>=0.25.0->peft) (0.16.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (75.2.0)
    Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (1.14.0)
    Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (3.6.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (3.1.6)
    Requirement already satisfied: cuda-bindings==12.9.4 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (12.9.4)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.8.93 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (12.8.93)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (12.8.90)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (12.8.90)
    Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (9.10.2.21)
    Requirement already satisfied: nvidia-cublas-cu12==12.8.4.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (12.8.4.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.3.3.83 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (11.3.3.83)
    Requirement already satisfied: nvidia-curand-cu12==10.3.9.90 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (10.3.9.90)
    Requirement already satisfied: nvidia-cusolver-cu12==11.7.3.90 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (11.7.3.90)
    Requirement already satisfied: nvidia-cusparse-cu12==12.5.8.93 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (12.5.8.93)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (0.7.1)
    Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (2.27.5)
    Requirement already satisfied: nvidia-nvshmem-cu12==3.4.5 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (3.4.5)
    Requirement already satisfied: nvidia-nvtx-cu12==12.8.90 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (12.8.90)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.8.93 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (12.8.93)
    Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (1.13.1.3)
    Requirement already satisfied: triton==3.6.0 in /usr/local/lib/python3.12/dist-packages (from torch>=1.13.0->peft) (3.6.0)
    Requirement already satisfied: cuda-pathfinder~=1.1 in /usr/local/lib/python3.12/dist-packages (from cuda-bindings==12.9.4->torch>=1.13.0->peft) (1.4.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch>=1.13.0->peft) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=1.13.0->peft) (3.0.3)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/dist-packages (from transformers->peft) (2025.11.3)
    Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /usr/local/lib/python3.12/dist-packages (from transformers->peft) (0.22.2)
    Requirement already satisfied: click>=8.2.1 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface_hub>=0.25.0->peft) (8.3.1)
    Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface_hub>=0.25.0->peft) (1.5.4)
    Requirement already satisfied: rich>=12.3.0 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface_hub>=0.25.0->peft) (13.9.4)
    Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface_hub>=0.25.0->peft) (0.0.4)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface_hub>=0.25.0->peft) (4.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface_hub>=0.25.0->peft) (2.19.2)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/dist-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->huggingface_hub>=0.25.0->peft) (0.1.2)
    Downloading peft-0.18.1-py3-none-any.whl (556 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m557.0/557.0 kB[0m [31m8.4 MB/s[0m  [33m0:00:00[0m
    [?25hInstalling collected packages: peft
      Attempting uninstall: peft
        Found existing installation: peft 0.7.0
        Uninstalling peft-0.7.0:
          Successfully uninstalled peft-0.7.0
    Successfully installed peft-0.18.1


## 데이터 준비 및 학습

- 데이터:
    - 클래스: 할리우드 남성 영화 배우 이미지 약 130장
    - 인스턴스: 그 중 Brad Pitt 이미지 6장

    - 남자 영화 배우의 실사 이미지와 Brad Pitt 이미지로 학습시켜, Brad Pitt가 담긴 나만의 새로운 이미지를 생성한다.


```python
script_content = """#!/bin/bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./drive/MyDrive/Aiffel/CelebrityFacesDataset/male/BradPitt"
export CLASS_DIR="./drive/MyDrive/Aiffel/CelebrityFacesDataset/male/etc"
export OUTPUT_DIR="./diffusers_git/examples/dreambooth/data"

echo $MODEL_NAME

accelerate launch ./diffusers_git/examples/dreambooth/train_dreambooth.py \\
  --pretrained_model_name_or_path=$MODEL_NAME  \\
  --instance_data_dir=$INSTANCE_DIR \\
  --class_data_dir=$CLASS_DIR \\
  --output_dir=$OUTPUT_DIR \\
  --instance_prompt="a photo of Brad Pitt actor" \\
  --class_prompt="a photo of actor" \\
  --resolution=512 \\
  --train_batch_size=2 \\
  --with_prior_preservation --prior_loss_weight=1.0 \\
  --gradient_accumulation_steps=1 --gradient_checkpointing \\
  --use_8bit_adam \\
  --enable_xformers_memory_efficient_attention \\
  --set_grads_to_none \\
  --learning_rate=2e-6 \\
  --lr_scheduler="constant" \\
  --lr_warmup_steps=0 \\
  --num_class_images=9 \\
  --max_train_steps=500
"""

with open("train_dreambooth.sh", "w") as f:
    f.write(script_content)

```


```python
!rm -rf ./diffusers_git/examples/dreambooth/dog/.cache
!sh ./train_dreambooth.sh

print('----'*77)
print('학습 완료!!')
```

    CompVis/stable-diffusion-v1-4
    Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.
    Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.
    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    You are using a model of type `clip_text_model` to instantiate a model of type ``. This may be expected if you are loading a checkpoint that shares a subset of the architecture (e.g., loading a `sam2_video` checkpoint into `Sam2Model`), but is otherwise not supported and can yield errors. Please verify that the checkpoint is compatible with the model you are instantiating.
    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_validators.py:202: UserWarning: The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`. Downloading to a local directory does not use symlinks anymore.
      warnings.warn(
    {'sample_max_value', 'thresholding', 'variance_type', 'dynamic_thresholding_ratio', 'clip_sample_range', 'rescale_betas_zero_snr', 'timestep_spacing', 'prediction_type'} was not found in config. Values will be initialized to default values.
    Loading weights: 100% 196/196 [00:00<00:00, 5158.92it/s]
    [1mCLIPTextModel LOAD REPORT[0m from: CompVis/stable-diffusion-v1-4
    Key                                | Status     |  | 
    -----------------------------------+------------+--+-
    text_model.embeddings.position_ids | [38;5;208mUNEXPECTED[0m |  | 
    
    [3mNotes:
    - [38;5;208mUNEXPECTED[0m[3m	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.[0m
    {'force_upcast', 'use_quant_conv', 'use_post_quant_conv', 'shift_factor', 'latents_std', 'mid_block_add_attention', 'latents_mean', 'norm_num_groups'} was not found in config. Values will be initialized to default values.
    All model checkpoint weights were used when initializing AutoencoderKL.
    
    All the weights of AutoencoderKL were initialized from the model checkpoint at CompVis/stable-diffusion-v1-4.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use AutoencoderKL for predictions without further training.
    {'resnet_skip_time_act', 'encoder_hid_dim', 'resnet_time_scale_shift', 'upcast_attention', 'class_embed_type', 'mid_block_only_cross_attention', 'encoder_hid_dim_type', 'time_embedding_act_fn', 'only_cross_attention', 'time_embedding_dim', 'addition_embed_type_num_heads', 'timestep_post_act', 'time_cond_proj_dim', 'reverse_transformer_layers_per_block', 'dual_cross_attention', 'class_embeddings_concat', 'num_class_embeds', 'time_embedding_type', 'addition_embed_type', 'resnet_out_scale_factor', 'transformer_layers_per_block', 'num_attention_heads', 'addition_time_embed_dim', 'conv_in_kernel', 'conv_out_kernel', 'attention_type', 'projection_class_embeddings_input_dim', 'cross_attention_norm', 'dropout', 'use_linear_projection', 'mid_block_type'} was not found in config. Values will be initialized to default values.
    All model checkpoint weights were used when initializing UNet2DConditionModel.
    
    All the weights of UNet2DConditionModel were initialized from the model checkpoint at CompVis/stable-diffusion-v1-4.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use UNet2DConditionModel for predictions without further training.
    2026-03-05 18:31:26.742042: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1772735486.772084   13244 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1772735486.781376   13244 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    W0000 00:00:1772735486.817255   13244 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1772735486.817296   13244 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1772735486.817300   13244 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    W0000 00:00:1772735486.817304   13244 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
    Steps: 100% 500/500 [38:02<00:00,  4.18s/it, loss=0.153, lr=2e-6]Configuration saved in ./diffusers_git/examples/dreambooth/data/checkpoint-500/unet/config.json
    Model weights saved in ./diffusers_git/examples/dreambooth/data/checkpoint-500/unet/diffusion_pytorch_model.safetensors
    Steps: 100% 500/500 [40:01<00:00,  4.18s/it, loss=0.107, lr=2e-6]{'image_encoder', 'requires_safety_checker'} was not found in config. Values will be initialized to default values.
    
    Loading pipeline components...:   0% 0/7 [00:00<?, ?it/s][A
    
    Loading weights:   0% 0/396 [00:00<?, ?it/s][A[A
    
    Loading weights:  19% 76/396 [00:00<00:00, 709.18it/s][A[A
    
    Loading weights:  38% 150/396 [00:00<00:00, 726.64it/s][A[A
    
    Loading weights:  58% 229/396 [00:00<00:00, 745.31it/s][A[A
    
    Loading weights:  77% 306/396 [00:00<00:00, 739.25it/s][A[A
    
    Loading weights: 100% 396/396 [00:00<00:00, 753.03it/s]
    [1mStableDiffusionSafetyChecker LOAD REPORT[0m from: /root/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/safety_checker
    Key                                               | Status     |  | 
    --------------------------------------------------+------------+--+-
    vision_model.vision_model.embeddings.position_ids | [38;5;208mUNEXPECTED[0m |  | 
    
    [3mNotes:
    - [38;5;208mUNEXPECTED[0m[3m	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.[0m
    Loaded safety_checker as StableDiffusionSafetyChecker from `safety_checker` subfolder of CompVis/stable-diffusion-v1-4.
    
    Loading pipeline components...:  14% 1/7 [00:00<00:04,  1.39it/s][ALoaded tokenizer as CLIPTokenizer from `tokenizer` subfolder of CompVis/stable-diffusion-v1-4.
    
    Loading pipeline components...:  43% 3/7 [00:00<00:00,  4.26it/s][A{'timestep_spacing', 'prediction_type'} was not found in config. Values will be initialized to default values.
    Loaded scheduler as PNDMScheduler from `scheduler` subfolder of CompVis/stable-diffusion-v1-4.
    {'force_upcast', 'use_quant_conv', 'use_post_quant_conv', 'shift_factor', 'latents_std', 'mid_block_add_attention', 'latents_mean', 'norm_num_groups'} was not found in config. Values will be initialized to default values.
    All model checkpoint weights were used when initializing AutoencoderKL.
    
    All the weights of AutoencoderKL were initialized from the model checkpoint at /root/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/vae.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use AutoencoderKL for predictions without further training.
    Loaded vae as AutoencoderKL from `vae` subfolder of CompVis/stable-diffusion-v1-4.
    
    Loading pipeline components...:  86% 6/7 [00:01<00:00,  6.71it/s][ALoaded feature_extractor as CLIPImageProcessor from `feature_extractor` subfolder of CompVis/stable-diffusion-v1-4.
    Loading pipeline components...: 100% 7/7 [00:01<00:00,  6.16it/s]
    {'timestep_spacing', 'prediction_type'} was not found in config. Values will be initialized to default values.
    Configuration saved in ./diffusers_git/examples/dreambooth/data/vae/config.json
    Model weights saved in ./diffusers_git/examples/dreambooth/data/vae/diffusion_pytorch_model.safetensors
    
    Writing model shards:   0% 0/1 [00:00<?, ?it/s][A
    Writing model shards: 100% 1/1 [00:13<00:00, 13.98s/it]
    Configuration saved in ./diffusers_git/examples/dreambooth/data/unet/config.json
    Model weights saved in ./diffusers_git/examples/dreambooth/data/unet/diffusion_pytorch_model.safetensors
    Configuration saved in ./diffusers_git/examples/dreambooth/data/scheduler/scheduler_config.json
    
    Writing model shards:   0% 0/1 [00:00<?, ?it/s][A
    Writing model shards: 100% 1/1 [00:26<00:00, 26.57s/it]
    Configuration saved in ./diffusers_git/examples/dreambooth/data/model_index.json
    Steps: 100% 500/500 [43:43<00:00,  5.25s/it, loss=0.107, lr=2e-6]
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    학습 완료!!


## 이미지 생성 파이프라인 구성


```python
# %reset -f

# 의존성 모듈을 삭제 후 다시 설치합니다.
!pip uninstall -y diffusers
!pip install diffusers

from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

# huggingface에 미리 등록된 base 모델을 다운로드하여 사용합니다.
model_id = "CompVis/stable-diffusion-v1-4"

# 앞서 학습 코드로 만들어진 파라미터들을 로드합니다.
unet = UNet2DConditionModel.from_pretrained("./diffusers_git/examples/dreambooth/data/unet")
text_encoder = CLIPTextModel.from_pretrained("./diffusers_git/examples/dreambooth/data/text_encoder")

# stable diffusion 의 전체 파이프라인을 구성해줍니다.
pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16)
pipeline.to("cuda")
```

    Found existing installation: diffusers 0.37.0
    Uninstalling diffusers-0.37.0:
      Successfully uninstalled diffusers-0.37.0
    Collecting diffusers
      Using cached diffusers-0.37.0-py3-none-any.whl.metadata (20 kB)
    Requirement already satisfied: importlib_metadata in /usr/local/lib/python3.12/dist-packages (from diffusers) (8.7.1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from diffusers) (3.24.3)
    Requirement already satisfied: httpx<1.0.0 in /usr/local/lib/python3.12/dist-packages (from diffusers) (0.28.1)
    Requirement already satisfied: huggingface-hub<2.0,>=0.34.0 in /usr/local/lib/python3.12/dist-packages (from diffusers) (1.5.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from diffusers) (2.0.2)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/dist-packages (from diffusers) (2025.11.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from diffusers) (2.32.4)
    Requirement already satisfied: safetensors>=0.3.1 in /usr/local/lib/python3.12/dist-packages (from diffusers) (0.7.0)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.12/dist-packages (from diffusers) (11.3.0)
    Requirement already satisfied: anyio in /usr/local/lib/python3.12/dist-packages (from httpx<1.0.0->diffusers) (4.12.1)
    Requirement already satisfied: certifi in /usr/local/lib/python3.12/dist-packages (from httpx<1.0.0->diffusers) (2026.2.25)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/dist-packages (from httpx<1.0.0->diffusers) (1.0.9)
    Requirement already satisfied: idna in /usr/local/lib/python3.12/dist-packages (from httpx<1.0.0->diffusers) (3.11)
    Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/dist-packages (from httpcore==1.*->httpx<1.0.0->diffusers) (0.16.0)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers) (2025.3.0)
    Requirement already satisfied: hf-xet<2.0.0,>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers) (1.3.1)
    Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers) (26.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers) (6.0.3)
    Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers) (4.67.3)
    Requirement already satisfied: typer in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers) (0.24.1)
    Requirement already satisfied: typing-extensions>=4.1.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers) (4.15.0)
    Requirement already satisfied: zipp>=3.20 in /usr/local/lib/python3.12/dist-packages (from importlib_metadata->diffusers) (3.23.0)
    Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->diffusers) (3.4.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->diffusers) (2.5.0)
    Requirement already satisfied: click>=8.2.1 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub<2.0,>=0.34.0->diffusers) (8.3.1)
    Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub<2.0,>=0.34.0->diffusers) (1.5.4)
    Requirement already satisfied: rich>=12.3.0 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub<2.0,>=0.34.0->diffusers) (13.9.4)
    Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub<2.0,>=0.34.0->diffusers) (0.0.4)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface-hub<2.0,>=0.34.0->diffusers) (4.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface-hub<2.0,>=0.34.0->diffusers) (2.19.2)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/dist-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->huggingface-hub<2.0,>=0.34.0->diffusers) (0.1.2)
    Using cached diffusers-0.37.0-py3-none-any.whl (5.0 MB)
    Installing collected packages: diffusers
    Successfully installed diffusers-0.37.0


    Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.
    Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.



    Loading weights:   0%|          | 0/196 [00:00<?, ?it/s]


    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    Keyword arguments {'dtype': torch.float16} are not expected by StableDiffusionPipeline and will be ignored.



    Loading pipeline components...:   0%|          | 0/7 [00:00<?, ?it/s]



    Loading weights:   0%|          | 0/396 [00:00<?, ?it/s]


    [1mStableDiffusionSafetyChecker LOAD REPORT[0m from: /root/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/safety_checker
    Key                                               | Status     |  | 
    --------------------------------------------------+------------+--+-
    vision_model.vision_model.embeddings.position_ids | UNEXPECTED |  | 
    
    [3mNotes:
    - UNEXPECTED[3m	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.[0m





    StableDiffusionPipeline {
      "_class_name": "StableDiffusionPipeline",
      "_diffusers_version": "0.37.0",
      "_name_or_path": "CompVis/stable-diffusion-v1-4",
      "feature_extractor": [
        "transformers",
        "CLIPImageProcessor"
      ],
      "image_encoder": [
        null,
        null
      ],
      "requires_safety_checker": true,
      "safety_checker": [
        "stable_diffusion",
        "StableDiffusionSafetyChecker"
      ],
      "scheduler": [
        "diffusers",
        "PNDMScheduler"
      ],
      "text_encoder": [
        "transformers",
        "CLIPTextModel"
      ],
      "tokenizer": [
        "transformers",
        "CLIPTokenizer"
      ],
      "unet": [
        "diffusers",
        "UNet2DConditionModel"
      ],
      "vae": [
        "diffusers",
        "AutoencoderKL"
      ]
    }



## 이미지 생성


```python
prompt = "A photo of Brad Pitt actor acting in a romance movie"
image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("brad-pitt-dreambooth.png")
image
```


      0%|          | 0/50 [00:00<?, ?it/s]





    
![png](output_29_1.png)
    



- 로맨스 영화를 찍고 있는 브래드 피트 이미지를 생성하도록 했다.
- 로맨스 영화여서 그런지, 브래드 피트의 실제 배우자이나 로맨스 영화에서 같이 연기한 적 있는 안젤리나 졸리의 이미지도 함께 생성됐다.
- 그러나, 키스를 하려는 장면으로 보이지만, 얼굴이 부자연스럽고 딱딱한 표정을 짓고 있는 점이 아쉽다.


```python
prompt = "A photo of Brad Pitt actor with other actors"
image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("brad-pitt-dreambooth2.png")
image
```


      0%|          | 0/50 [00:00<?, ?it/s]





    
![png](output_31_1.png)
    



- 다양한 배우들과 함께 있는 브래드 피트 이미지를 생성하도록 하여 브래드 피트와 다른 배우들을 잘 구분하여 그려지는 지 확인하였다.
- 마치 시상식의 포토월에서 찍힌 사진처럼 이미지가 생성되었다.
- 그러나, 얼굴이 뭉개지거나 상당히 부자연스럽게 그려졌으며, 브래드 피트처럼 보이는 인물이 다수 존재했다.

## Hyperparameter
- 얼굴이 담긴 이미지의 경우 Hyperparameter를 변경함으로써 더 자연스러운 이미지를 생성할 수 있다. 따라서 아래와 같이 hyperparmeter를 변경하여 finetuning을 수행했다.
  - max_train_steps: 100 -> 500
  - train_batch_size: 1 -> 2
- 추가적으로 num_inference_steps: 50 -> 100 으로 하여 더 많은 추론 스텝을 수행하도록 했다.


```python
prompt = "A photo of Brad Pitt actor acting in a romance movie"
image = pipeline(prompt, num_inference_steps=100, guidance_scale=7.5).images[0]

image.save("/content/drive/MyDrive/brad-pitt-dreambooth_h1.png")
image
```


      0%|          | 0/100 [00:00<?, ?it/s]





    
![png](output_34_1.png)
    




```python
prompt = "A photo of Brad Pitt actor with other actors"
image = pipeline(prompt, num_inference_steps=100, guidance_scale=7.5).images[0]

image.save("/content/drive/MyDrive/brad-pitt-dreambooth_h2.png")
image
```


      0%|          | 0/100 [00:00<?, ?it/s]





    
![png](output_35_1.png)
    



hyperparamet를 수정하더라도 생성되는 이미지의 품질이 좋지 않앗다.
- romance movie에 대한 prompting을 부여했음에도 추가적인 여배우 이미지만 생성되지 로맨스 영화를 연기중이라는 모습이라고 생각할 수 없다.
- 다른 actor들과의 이미지를 생성할 때에는 여전히 부자연스러운 얼굴을 생성했다.

### 이미지 생성이 guidance_scale 적용 결과


```python
prompt = "A photo of Brad Pitt actor with other actors"
image = pipeline(prompt, num_inference_steps=50, guidance_scale=3).images[0]

image.save("brad-pitt-dreambooth4.png")
image
```


      0%|          | 0/50 [00:00<?, ?it/s]





    
![png](output_38_1.png)
    



- guidance_scale 값을 7.5에서 3으로 낮추자 브래드 피트 두명과 한명의 여배우 이미지가 생성되었다.
- 여러 배우들과 같이 있는 이미지라기 보다 별개의 상황에서 찍힌 사진을 이어 붙인 이미지가 생성되었다.



```python
prompt = "A photo of Brad Pitt actor with other actors"
image = pipeline(prompt, num_inference_steps=50, guidance_scale=10).images[0]

image.save("brad-pitt-dreambooth5.png")
image
```


      0%|          | 0/50 [00:00<?, ?it/s]





    
![png](output_40_1.png)
    



- guidance_scale 값을 10으로 올리자 브래드 피트의 얼굴이 다소 비현실적으로 변화했다.

- guidance_scale을 눞였을 때, 최대한 프롬프트에서 요구한 내용을 지키는 것으로 알고 잇지만, 여전히 다른 배우들이 아닌 브래드 피트에 대한 이미지만 생성하고 있다.

# Stable diffusion 모델을 자유롭게 요모조모 다뤄보기

## LoRA 적용해보기


```python
!pip install diffusers

```

    Requirement already satisfied: diffusers in /usr/local/lib/python3.12/dist-packages (0.36.0)
    Requirement already satisfied: importlib_metadata in /usr/local/lib/python3.12/dist-packages (from diffusers) (8.7.1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from diffusers) (3.24.3)
    Requirement already satisfied: httpx<1.0.0 in /usr/local/lib/python3.12/dist-packages (from diffusers) (0.28.1)
    Requirement already satisfied: huggingface-hub<2.0,>=0.34.0 in /usr/local/lib/python3.12/dist-packages (from diffusers) (1.5.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from diffusers) (2.0.2)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/dist-packages (from diffusers) (2025.11.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from diffusers) (2.32.4)
    Requirement already satisfied: safetensors>=0.3.1 in /usr/local/lib/python3.12/dist-packages (from diffusers) (0.7.0)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.12/dist-packages (from diffusers) (11.3.0)
    Requirement already satisfied: anyio in /usr/local/lib/python3.12/dist-packages (from httpx<1.0.0->diffusers) (4.12.1)
    Requirement already satisfied: certifi in /usr/local/lib/python3.12/dist-packages (from httpx<1.0.0->diffusers) (2026.2.25)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/dist-packages (from httpx<1.0.0->diffusers) (1.0.9)
    Requirement already satisfied: idna in /usr/local/lib/python3.12/dist-packages (from httpx<1.0.0->diffusers) (3.11)
    Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/dist-packages (from httpcore==1.*->httpx<1.0.0->diffusers) (0.16.0)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers) (2025.3.0)
    Requirement already satisfied: hf-xet<2.0.0,>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers) (1.3.1)
    Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers) (26.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers) (6.0.3)
    Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers) (4.67.3)
    Requirement already satisfied: typer in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers) (0.24.1)
    Requirement already satisfied: typing-extensions>=4.1.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2.0,>=0.34.0->diffusers) (4.15.0)
    Requirement already satisfied: zipp>=3.20 in /usr/local/lib/python3.12/dist-packages (from importlib_metadata->diffusers) (3.23.0)
    Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->diffusers) (3.4.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->diffusers) (2.5.0)
    Requirement already satisfied: click>=8.2.1 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub<2.0,>=0.34.0->diffusers) (8.3.1)
    Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub<2.0,>=0.34.0->diffusers) (1.5.4)
    Requirement already satisfied: rich>=12.3.0 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub<2.0,>=0.34.0->diffusers) (13.9.4)
    Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub<2.0,>=0.34.0->diffusers) (0.0.4)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface-hub<2.0,>=0.34.0->diffusers) (4.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface-hub<2.0,>=0.34.0->diffusers) (2.19.2)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/dist-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->huggingface-hub<2.0,>=0.34.0->diffusers) (0.1.2)


https://civitai.com/images/122961814


```python
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained("martineux/waiIllustriousSDXL_v160", torch_dtype=torch.float16)   # 알맞은 모델 ID 를 입력합니다.
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.to("cuda")

pipeline.load_lora_weights("/content/drive/MyDrive/ppw_v8_Illuv2stable_128.safetensors")   # 다운로드한 LoRA 를 로드합니다.
```

    Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.
    Flax classes are deprecated and will be removed in Diffusers v1.0.0. We recommend migrating to PyTorch classes or pinning your version of Diffusers.
    /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(



    model_index.json:   0%|          | 0.00/712 [00:00<?, ?B/s]



    Downloading (incomplete total...): 0.00B [00:00, ?B/s]



    Fetching 18 files:   0%|          | 0/18 [00:00<?, ?it/s]


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.



    Loading pipeline components...:   0%|          | 0/7 [00:00<?, ?it/s]



    Loading weights:   0%|          | 0/196 [00:00<?, ?it/s]



    Loading weights:   0%|          | 0/517 [00:00<?, ?it/s]



    Loading weights:   0%|          | 0/144 [00:00<?, ?it/s]



    Loading weights:   0%|          | 0/384 [00:00<?, ?it/s]



```python
image = pipeline(
    prompt="masterpiece, best quality, very aesthetic, CivitAI BatouLaMenace, snow capped mountain peak, alpine valley, pine trees, distant mountain range, no humans, wide landscape shot, elevated viewpoint, leading lines, dramatic perspective, golden hour, sunset sky, scattered clouds, warm sunlight,",
    negative_prompt="bad quality, worst quality, worst detail, sketch, censor, signature, watermark, text,",
    num_inference_steps=28,
    guidance_scale=7,
).images[0]

image.save("/content/drive/MyDrive/sd_lora_sample1.png")
image
```


      0%|          | 0/28 [00:00<?, ?it/s]





    
![png](output_47_1.png)
    



- StableDiffusionXL 모델을 활용하였다.
- prompt의 token 수가 77개를 넘어가면 prompt의 일부가 반영되지 않기에 prompt 길이를 줄여서 사용했다.
- 눈으로 덮인 산과 해질녘 노을로 이루어진 이미지가 생성됐다.

- 협곡이 이어지는 중간 부분이 부자연스러운 점이 아쉽다.


```python
image = pipeline(
    prompt="masterpiece, best quality, very aesthetic, CivitAI BatouLaMenace, snow capped mountain peak, alpine valley, pine trees, distant mountain range, no humans, wide landscape shot, elevated viewpoint, leading lines, dramatic perspective, shooting star, night sky, scattered clouds, cold moonlight,",
    negative_prompt="bad quality, worst quality, worst detail, sketch, censor, signature, watermark, text,",
    num_inference_steps=28,
    guidance_scale=7,
).images[0]

image.save("/content/drive/MyDrive/sd_lora_sample2.png")
image
```


      0%|          | 0/28 [00:00<?, ?it/s]





    
![png](output_49_1.png)
    



- 해질녘 하늘 대신, 별똥별과 달빛이 비춰지는 눈 덮인 산 이미지를 생성하도록 했다.

- 오로라는 prompt로 주어지지 않았지만, 이미지에 생성이 되었고, 차가운 밤 하늘이 보여지도록 prompt를 입력했지만 여전히 노을 빛이 남아있는 점이 아쉽다.


```python

```
