def test_custom_ops():
    # 初始化模型和其他必要组件
    model = initialize_model()
    data = load_test_data()

    # 执行前向传播
    outputs = model(data)
    
    # 验证输出的形状和范围是否正确
    assert outputs.shape == expected_shape, "输出形状不匹配"
    assert (outputs >= min_value).all() and (outputs <= max_value).all(), "输出值超出预期范围"

    # 执行反向传播并验证梯度
    loss = compute_loss(outputs)
    gradients = torch.autograd.grad(loss, model.parameters())
    for grad in gradients:
        assert not torch.isnan(grad).any(), "梯度包含NaN"
        assert not torch.isinf(grad).any(), "梯度包含无穷大"

    # 额外的验证步骤
    validate_custom_ops(model)