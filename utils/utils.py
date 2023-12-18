def update_module_step(m, epoch, global_step):
    if hasattr(m, "update_step"):
        m.update_step(epoch, global_step)
