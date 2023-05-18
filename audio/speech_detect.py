# import speech_recognition as sr
import pyttsx3
import openai
import pocketsphinx
from pocketsphinx import LiveSpeech, get_model_path
import os
import pyautogui
from gtts import gTTS
import playsound
import subprocess
import AppOpener
from threading import Thread

# Module for communicating with the Google API fro speech recognition
from . import google_speech_recognition as gsr
# import google_speech_recognition as gsr

# import PyAudio
class SpeechDetect():
    def __init__(self):
        openai.api_key = "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"

        self.cursor_enable = False
        self.cursor_speed_ratio = 1.0

        audio_tread = Thread(target=self.audio_init, args=())
        audio_tread.start()

    def SpeakText(self,command):
        # Initialize the engine
        engine = pyttsx3.init()
        engine.say(command)
        engine.runAndWait()


    def open_application(self,text):
        success = True
        print(f'opening {text}...')
        AppOpener.open(text, match_closest=True)


    def get_gpt_response(self,text):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt='You said: ' + text + "\n",
            max_tokens=2048,
            temperature=0.5
        )
        response_text = response.choices[0].text
        return response_text


    def control(self,text):
        control_mode = True
        if "click" in text:
            pyautogui.click()
        elif "up" in text:
            pyautogui.moveRel(0, -10)
        elif "down" in text:
            pyautogui.moveRel(0, 10)
        elif "left" in text:
            pyautogui.moveRel(-10, 0)
        elif "right" in text:
            pyautogui.moveRel(10, 0)
        elif "stop" in text:
            pyautogui.moveRel(0, 0)
            control_mode = False
        elif "cursor" in text:
            self.cursor_enable = not self.cursor_enable
        elif "fast" in text:
            self.cursor_speed_ratio = 1.0
        elif "slow" in text:
            self.cursor_speed_ratio = 0.5
        return control_mode


    def set_mode(self,text, mode_list):
        mode = mode_list[0]

        for i in range(len(mode_list)):
            if mode_list[i] == text:
                mode = mode_list[i]
                print("SET")
        return mode

    def audio_init(self):
            # MODELDIR = '/Users/jacksonwagner/opt/anaconda3/envs/my_tf_env/lib/python3.9/site-packages/pocketsphinx/model/'
        # MODELDIR = get_model_path()
        # config = pocketsphinx.Decoder.default_config()
        # config.set_string('-hmm', os.path.join(MODELDIR, 'en-us/en-us'))
        # config.set_string('-lm', os.path.join(MODELDIR, 'en-us/en-us.lm.bin'))
        # config.set_string('-dict', os.path.join(MODELDIR, 'en-us/cmudict-en-us.dict'))
        # decoder = pocketsphinx.Decoder(config)

        # modes: control, dictation, chatbot, open_app, listen
        mode_list = ['listen', 'control', 'chat', 'applications', 'dictate']
        mode = mode_list[0]

        control_command_list = ['click', 'up', 'down', 'left', 'right', 'stop', 'cursor','fast','slow']

        applications_list = ['']

        listen = True

        while listen:
            # workflow if the user has not selected a mode yet (still in listen mode)
            if mode == mode_list[0]:
                self.SpeakText('Select a mode. You can say things like control, dictate, chat, or applications')
                print('Select a mode. You can say things like control, dictate, chat, or applications')
                text = gsr.start_listen(mode="select_mode")

                if text == 'exit':
                    break

                mode = self.set_mode(text, mode_list)
                
                # while mode == mode_list[0]:
                #     SpeakText(
                #         'Sorry, I did not catch that. You can say things like control, dictate, chat, or applications')
                #     print('Sorry, I did not catch that. You can say things like control, dictate, chat, or applications')
                #     text = listen_and_return_mode()
                #     mode = set_mode(text, mode_list)


            # control mode 
            elif mode == mode_list[1]:
                self.SpeakText('Control mode activated. Say things like cursor, fast, slow.')
                print('Control mode activated. Say things like cursor, fast, slow.')
                # run the control loop where we execute a command from the user until we get a new command                
                control_mode = True
                while control_mode:
                    text = gsr.start_listen(mode="text")
                    
                    if 'exit' in text:
                        break

                    for c in control_command_list:
                        if c in text:
                            control_mode = self.control(text)

                # once the user says stop, we reset the mode to listen mode
                mode = mode_list[0]


            # chat mode
            elif mode == mode_list[2]:
                self.SpeakText('Chat mode activated. Ask a question or say done to go to a different mode.')
                print('Chat mode activated. Ask a question or say done to go to a different mode.')

                chat = True
                while chat:
                    text = gsr.start_listen(mode="text")

                    if 'exit' in text:
                        break
                    else:
                        response = self.get_gpt_response(text)
                        print('chat bot response: ', response)
                        self.SpeakText(response)

                mode = mode_list[0]

            # applications mode
            elif mode == mode_list[3]:
                self.SpeakText(
                    'Applications mode activated. You can open applications by saying things like chrome, safari, finder, documents, zoom, messages, calendar.')
                print(
                    'Applications mode activated. You can open applications by saying things like chrome, safari, finder, documents, zoom, messages, calendar.')
                # text = gsr.start_listen(mode="text")
                # self.open_application(text)
                success = False
                while not success:
                    text = gsr.start_listen(mode="text")
                    if 'exit' in text:
                        break
                    else:
                        success = self.open_application(text)
                        if success == False:
                            self.SpeakText('application not found. try again or say end to change modes')
                            print('application not found. try again or say end to change modes')
                
                mode = mode_list[0]

            # dictation mode
            elif mode == mode_list[4]:
                self.SpeakText('Dictate mode activated. You can click on any input box and start speaking to write text.')
                print('Dictate mode activated. You can click on any input box and start speaking to write text.')
                
                # gsr.dictate()
                dictate = True
                while dictate:
                    text = gsr.start_listen(mode="dictate")

                    if 'exit' in text:
                        break
                    else:
                        pyautogui.typewrite(text)

                mode = mode_list[0]

if __name__ == "__main__":
    SpeechDetect().audio_init()
