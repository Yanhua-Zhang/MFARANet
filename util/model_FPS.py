import torch
import time

# ===========================================================
# 计算方式来自于 E:\Code_Python\PYTHON_code\Semantic_segmentation\3_SFNet、SFSegNets\utils\misc.py
def speed_test(model, size=896, iteration=100):
    input_t = torch.Tensor(1, 3, size, size).cuda()

    print("start warm up")

    for i in range(10):
        model(input_t)

    print("warm up done")
    start_ts = time.time()
    for i in range(iteration):
        model(input_t)

    torch.cuda.synchronize()
    end_ts = time.time()

    t_cnt = end_ts - start_ts
    print("=======================================")
    print("FPS: %f" % (100 / t_cnt))
    print(f"Inference time {t_cnt/100*1000} ms")

# 我的改进版本
def FPS_counter(model, model_name, height, weight, logger, iteration=100): 
    logger.info('model name：'+model_name+': ')
    logger.info('开始利用 SFNet 中方式计算 FPS：')

    # input_t = torch.Tensor(1, 3, height, weight).cuda()  # 会导致：RuntimeError: cuda runtime error (700) : an illegal memory access was encountered at
    input_t = torch.rand(1, 3, height, weight).cuda()
    model.eval()
    model.cuda()

    logger.info("start warm up")
    for i in range(10):
        model(input_t)
    logger.info("warm up done")

    start_ts = time.time()
    for i in range(iteration):
        model(input_t)

    torch.cuda.synchronize() # Waits for all kernels in all streams on a CUDA device to complete.
    end_ts = time.time()

    t_cnt = end_ts - start_ts
  
    logger.info("FPS: %f" % (100 / t_cnt))
    logger.info(f"Inference time {t_cnt/100*1000} ms")
    logger.info('End')
    logger.info('--------------------------------')
    logger.info('        ')

# ===========================================================
# 论文 swiftnet 提供的计算方式：
def swiftnet_counter():
    device = torch.device('cuda')
    model.eval()
    model.to(device)
    with torch.no_grad():
        input = model.prepare_data(batch).to(device)
        logits = model.forward(input)
        torch.cuda.synchronize()
        t0 = perf_counter()
        for _ in range(n):
            input = model.prepare_data(batch).to(device)
            logits = model.forward(input)
            _, pred = logits.max(1)
            out  = pred.data.byte().cpu()
        torch.cuda.synchronize()
        t1 = perf_counter()
        fps = n/(t1-t0)