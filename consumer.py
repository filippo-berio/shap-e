#!/usr/bin/env python
import os
import pika
import sys
import json

from src.store_file import store_file
from src.make_3d import make_model


def main():
    pika_params = pika.ConnectionParameters(
        host='localhost',
        credentials=pika.credentials.PlainCredentials('root', 'root')
    )

    connection = pika.BlockingConnection(pika_params)

    channel = connection.channel()
    channel.queue_declare(queue='shop_3d', durable=True)

    response_channel = connection.channel()
    response_channel.queue_declare(queue='shop_3d_result', durable=True)

    def callback(ch, method, properties, body):
        body = json.loads(body.decode('ASCII'))
        print(f'Received message: {body}. Creating 3D model...')

        images = body.get('images')
        model = make_model(images)
        print('Model created, storing...')

        model_url = store_file(model, body.get('uid'))
        print(f'Model stored at {model_url}')

        response = json.dumps({
            'result': model_url,
            'uid': body.get('uid')
        }).encode('ASCII')
        channel.basic_publish(exchange='', routing_key='shop_3d_result', body=response)

    channel.basic_consume(queue='shop_3d', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
