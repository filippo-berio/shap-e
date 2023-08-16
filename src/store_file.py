from minio import Minio


def store_file(file, uid):
    client = Minio('localhost:9000', access_key='root', secret_key='rootroot', secure=False)
    if not client.bucket_exists('shop-3d'):
        client.make_bucket('shop-3d')
    client.put_object(
        'shop-3d', f'models/{uid}.glb', file, -1, part_size=1 * 8 * 1024 * 1024
    )
    return f'localhost:9000/shop-3d/models/{uid}.glb'
