from config import *
import telebot
import threading
import watchdog.events
import watchdog.observers
import time

#instanciamos el bot
bot = telebot.TeleBot(TELEGRAM_TOKEN) 

user_ids = {}

last_alert_time = -1

left_image_path = '/home/nvidia/DomoEstatico/DomoJetson/left_image.jpg'
right_image_path = '/home/nvidia/DomoEstatico/DomoJetson/right_image.jpg'

log_path = "/home/nvidia/DomoEstatico/DomoJetson/logDomo.txt"

@bot.message_handler(content_types=["text"])
def bot_mensajes_texto(message):

    global left_image_path, right_image_path
    comando = message.text.strip().lower()

    if comando == "/start":
        user_id = message.chat.id
        user_ids[user_id] = True
        bot.reply_to(message, "Hola! Soy DomoBot")
    elif comando == "/log":
        #gestiona los msj de texto recibidos. Enviamos documento con distancias.
        bot.send_chat_action(message.chat.id, "upload_document")
        archivo = open(log_path, "rb")
        bot.send_document(message.chat.id, archivo, caption="Log de distancias obtenidas")
    elif comando == "/foto":
        # Lógica para obtener y enviar la imagen en tiempo real
        # bot.send_chat_action(message.chat.id, "upload_image")
        bot.send_photo(message.chat.id, photo=open(right_image_path, 'rb'))
        bot.send_photo(message.chat.id, photo=open(left_image_path, 'rb'))
    else:
        bot.send_message(message.chat.id, "Ingrese un comando valido: \n- /start -> iniciar comunicacion\n- /log -> archivo de registro de distancias\n- /foto -> imagen en tiempo real")
     

def recibir_mensajes():
    bot.infinity_polling()

# Función para enviar mensaje de alerta
def enviar_alerta(user_id):

    global left_image_path, right_image_path

    message = "¡Alerta! Se ha detectado una persona a menos de 2 metros."

    # Send the alert message and the images
    bot.send_message(user_id, message)
    bot.send_message(CHATID_CHANNEL, message)

    bot.send_photo(user_id, photo=open(left_image_path, 'rb'))
    bot.send_photo(user_id, photo=open(right_image_path, 'rb'))

    bot.send_photo(CHATID_CHANNEL, photo=open(left_image_path, 'rb'))
    bot.send_photo(CHATID_CHANNEL, photo=open(right_image_path, 'rb'))


# Clase para el observador de cambios en el archivo log
class LogEventHandler(watchdog.events.PatternMatchingEventHandler):
    def init(self):
        super().init(patterns=["*.txt"])

    def on_modified(self, event):
        global last_alert_time  # Declarar la variable como global
        if not event.is_directory:
            # Obtener el path del archivo modificado
            file_path = event.src_path

            # Obtener el tiempo actual
            current_time = time.time()

            # Verificar si ha transcurrido al menos 1 minuto desde la última alerta o es la primera alerta
            if last_alert_time < 0 or current_time - last_alert_time >= 15:

                for user_id in user_ids.keys():
                    enviar_alerta(user_id)
                last_alert_time = current_time
                

#Función para iniciar el observador de cambios en el archivo log
def iniciar_observador():
    event_handler = LogEventHandler()
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path="/home/nvidia/DomoEstatico/DomoJetson", recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except Exception:
        observer.stop()
    observer.join()

################### main ###################

if __name__ == '__main__':
    bot.set_my_commands([
        telebot.types.BotCommand("/start", "Iniciar comunicacion con el bot"),
        telebot.types.BotCommand("/log", "Obtener ultima medicion realizada"),
        telebot.types.BotCommand("/foto", "Obtener ultima imagen capturada")
    ])
    print('iniciando el bot')
    thread_bot = threading.Thread(name= "thread_bot", target=recibir_mensajes)     #bucle infinito que chequea si se reciben comandos nuevos
    thread_bot.start()

    # Iniciar el observador de cambios en el archivo log
    thread_observer = threading.Thread(name="thread_observer", target=iniciar_observador)
    thread_observer.start()

    bot.send_message(CHATID_CHANNEL, "¡El domoBot se encuentra activo!")



    
