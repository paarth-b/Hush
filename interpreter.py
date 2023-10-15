from recogCV import output_string
import pygame.mixer
import time


def play_sound_for_letter(letter):
    pygame.mixer.init()

    sound_path = f"sounds/{letter}.wav"

    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play()

    # Wait until the sound is done playing
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)


def main():
    for letter in output_string:
        if "A" <= letter <= "Z":  # Ensure the character is an uppercase letter
            play_sound_for_letter(letter)


if __name__ == "__main__":
    main()
