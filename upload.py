import modal

vol = modal.Volume.from_name("model_2", create_if_missing=True)
with vol.batch_upload() as batch:
    batch.put_directory("model_2/", "/model/")