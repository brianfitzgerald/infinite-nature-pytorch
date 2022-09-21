import torch
  
def as_numpy(x: torch.Tensor):
    if type(x) == torch.Tensor:
        return x.numpy()
    else:
        return x

  
  def render_refine(image, style_noise, pose, intrinsic,
                    pose_next, intrinsic_next):
    return sess.run(generated_image, feed_dict={
        image_placeholder: as_numpy(image),
        style_noise_placeholder: as_numpy(style_noise),
        pose_placeholder: as_numpy(pose),
        intrinsic_placeholder: as_numpy(intrinsic),
        pose_next_placeholder: as_numpy(pose_next),
        intrinsic_next_placeholder: as_numpy(intrinsic_next),
    })

  def encoding_fn(encoding_image):
    return sess.run(z, feed_dict={
        encoding_placeholder: as_numpy(encoding_image)})

  return render_refine, encoding_fn
