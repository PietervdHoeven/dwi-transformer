from monai.networks.nets import AutoEncoder

def get_monai_ae(channels=(16,32,64), strides=(2,2,2), num_res_units=2):
    return AutoEncoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
    )
