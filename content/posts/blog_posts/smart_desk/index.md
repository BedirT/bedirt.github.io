---
author: "Bedir Tapkan"
title: "Smarter Autonomous Desk: A Journey - Part 1"
date: 2024-12-17
description: "My journey to make my Autonomous branded desk smarter."
tags: ["Home Automation", "Smart Desk", "IoT", "Automation", "ESP", "Home Assistant"]
ShowToc: true
---

{{< image_centered
    src="https://img.shields.io/badge/Project%20Repository%20-%20SmartAutonomous%20-%20grey?style=flat&logo=github"
    alt="Github Badge"
    class="custom-class"
    link="https://github.com/BedirT/SmartAutonomous"
>}}


A little less than a year ago, I realized that I was spending most of my life in front of my desk and work setup. Especially since I started working from home (Thanks COVID...) this was always the same desk; my trusty IKEA desk with manual height adjustment. Since I no longer was a student, and had some disposable income, I decided to upgrade my desk to a Standing Desk that I anticipated would add more flexibility and movement to my setup, eventually enhancing my productivity and health.

![My Desk](./my_old_desk.jpg)

After some research and consideration, I decided to go with the Autonomous SmartDesk 2. This is not an advert, but I am happy with it so far. The good part that hooked me was that I could get the frame without the tabletop, which was important for me because I wanted a nice and wide table to house my work, personal and design setup. So I grabbed a large butcherblock from IKEA, and mounted it on the Autonomous frame. Hurray, I had a new smart (!) desk.

![My Desk](./my_new_desk.jpg)

One issue came to me over the months. The desk was not smart, I was supposed to be. The standing desk gives me the ability to adjust the height of the desk throughout the day, but again the ability is in your hands. So as long as you ignore that you have a standing desk, you can sit there all day long. I believe this is a common problem for my fellow standing desk owners, who purchased with a moments haste, and then realized that they are not using the desk as intended.

Having read a recent paper that standing was even not a solution, but a step towards it, I decided to make my desk actually smart, and take the guesswork and the manual labor of thinking out of the equation. I also wanted to gamify the process a bit to increase engagement and make it fun. As addition I also purchased a desk treadmill, which I will cover in a separate post.

Before starting this post, I must say that I had '0' experience with IoT, Home Automation, or any related field. I am experienced in coding and software development, but I had no idea about the hardware side of things. So this was a journey for me, and after seeing the lack of very beginner friendly guides, I decided to write this post to help others who are in the same boat as me.

## The Plan

The plan was simple. Making the desk smart, having it adjust based on certain rules. My immediate idea was to have a predefined schedule where the desk could just adjust itself. And to have the things more interesting, I also wanted to collect data about my standing/sitting habits, and maybe even gamify the process a bit. But without getting ahead of myself (since i literally have no idea what I am doing), I decided to break the project into smaller steps.

The initial version of the project would have the following features:

- **Automatic Desk Adjustment**: The desk would adjust itself based on a predefined schedule.
- **Data Collection**: The desk would collect data about the standing/sitting habits of the user.
- **Manual Adjustment**: The user would still be able to manually adjust the desk with the keypad on the desk.
- **Home Assistant Integration**: The desk would be integrated with Home Assistant for further automation and control.

## The Research

As much as I enjoy and love reinventing the wheel, I decided to do some research before starting the project. I wanted to see if there were any existing solutions that I could use, or at least get inspired by. I also wanted to see what kind of hardware I would need for the project.

Without too much trouble, I found a few projects that do automate the Autonomous Desk, yay! These had a sea of information for me; reverse engineered protocols, hardware recommendations, and full fetched systems even. Special thanks to the Stefichen5 for his write up and repo:

- [Stefichen5 - AutonomousControl](https://github.com/Stefichen5/AutonomousControl)

Brilliant project that reverse engineered the Autonomous Desk protocol and created a full-fledged C++ system around it. When all went to hell, this was my saviour! The code wasn't usable for me, but the protocol information was invaluable, and I also read the code to understand the protocol and the system better.

There are also these, but i didn't need to refer to them much:

- [at-karan-sharma - autonomous-desk-esphome](https://github.com/at-karan-sharma/autonomous-desk-esphome)

This project was a Home Assistant integration for the Autonomous Desk. I didn't get to use this because I failed to properly run it and by the time I knew what I was doing I had already written my own integration.

- [vipial1 - desky-ha](https://github.com/vipial1/desky-ha)
- [developit - desky](https://github.com/developit/desky)

The reason I didn't use these was that they were not as feature rich as I wanted them to be. I wanted to have a more flexible system that I could expand upon in the future. Apart from that, I also wanted to explore all the nooks and crannies of the project, and writing my own code was the best way to do that. And last reason is that none of these have a complete and detailed guide or documentation, which is a must for a person like me who is clueless.

## The Hardware

All the projects I found were using an ESP32 or an ESP8266 to control the desk. People used an external 10P10C connector add ESP to the desk controller. I thought that using an external connector was unnecessary, and I could just tap into the existing controller cable, add an ESP to it.

So I first decided to use ESP8266 because it's just cheaper :) I bought some NodeMCU ESP8266s to get started. I had a ELEGOO electronics starter kit that I bought a while ago, and I had some basic components like a breadboard, resistors, and jumper cables. Since I was winging and learning as I go, I bought some stuff I didn't need as well (like a logic level converter, which I didn't use in the end). I blame ChatGPT for guiding me to buy unnecessary stuff :)

After failing a countless times using ESP8266, I switched to ESP32 thinking my voltage level had something to do with the failure. In the end I made both of these work but there is a super minor difference so I explain everything with ESP32 in mind and will mention the difference at the end of the post.

Apart from these, I also purchased some Wago connectors to make sure my issues were not due to cable connections. I got a nice wire stripper because why not! I also got some wires and soldering equipment to finalize the connections permanently after I was done with the prototyping. One other thing I used was a multimeter to check the voltages and connections, but this is not necessary if you are not planning to do any testing. Lastly, we need a diode to be able to use the desk's keypad and the ESP at the same time. I used a 1N4148 diode, but any small signal diode should work (I tested with a 1N4007 as well and it works fine).

I also printed an enclosure with my 3D printer, but that is not necessary (though it looks way better and cleaner).

So here are final list of hardware I used in two categories:
1. **Used in final product**:
    - ESP32 (Wroom Esp32) / ESP8266 (NodeMCU)
    - Diode (1N4148)
    - 24 AWG Wires
    - Soldering Iron
    - Solders
    - Enclosure

2. **Used in prototyping and testing**:
    - Breadboard
    - Jumper Cables
    - Wago Connectors
    - Wire Stripper (Optional)
    - Multimeter (Optional)

Couple videos and tutorials I used to get started with the hardware side of things:
- [Diodes Explained - The basics how diodes work working principle pn junction](https://www.youtube.com/watch?v=Fwj_d3uO5g8)
- [Soldering Tutorial for Beginners: Five Easy Steps](https://www.youtube.com/watch?v=Qps9woUGkvI)
- [HOW TO SOLDER! (Beginner's Guide)](https://www.youtube.com/watch?v=3jAw41LRBxU)
- [Arduino To ESP32: How to Get Started!](https://www.youtube.com/watch?v=RiYnucfy_rs)
- [A beginner’s guide to ESP32 | Hardware & coding basics + Wi-Fi server demo](https://www.youtube.com/watch?v=UuxBfKA3U5M)
- [ESP8266 in 5 minutes](https://www.youtube.com/watch?v=dGrJi-ebZgI)

## The Software

Given that I wanted to end up with a Home Assistant integration, I first started with esphome. Esphome is a great tool that allows you to create custom firmware for ESP devices. It has a nice and easy to use YAML configuration that you can use to define your devices and their behavior. I started with this because I thought it would be the easiest way to get started, and I could easily integrate it with Home Assistant. But having the entire system new to me, I got discourged and switched to MicroPython and decided that after figuring our all the details I would switch back and implement a version with esphome as well.

Micropython is a Python implementation for microcontrollers. It is a great tool that allows you to write Python code and run it on microcontrollers. I chose this because I am very comfortable with Python, and I thought it would be easier for me to get started with this. Apparently this also gives more freedom and control over the system, which I thought would be beneficial for me in the long run. I used Thonny IDE to write and run my code on the ESP32.

After finishing with the prototyping and testing, I wanted to move the micropython implementation to Home Assistant. I then met with MQTT, which is a messaging protocol that allows you to send messages between devices. I used this to send messages between the ESP32 and Home Assistant. After finalizing MQTT integration, for practice and learning purposes, I went back to esphome and implemented a version with it. This was a great learning experience for me, and I am happy that I did it.

In this post we are only focusing on the Micropython implementation. I will write a separate post for the esphome implementation.

## The Protocol

A protocol is a set of rules that defines how devices communicate with each other. In order to control the desk, we need to know the protocol that the desk controller uses. This is the way that the keypad communicates with the desk controller.

This is where I got most of the information from Stefichen5's project. There were still some missing parts, and stuff that I couldn't understand and needed to test, which I did with a series of trial and error.

Understanding the communication protocol used by the desk's control system was the most challenging yet rewarding part of the project. The desk communicates using a simple UART-based protocol, where messages are exchanged between the keypad and the motor controller. To successfully control the desk, we had to mimic these messages and decipher their meaning.

[Tutorial on UART](https://www.youtube.com/watch?v=IyGwvGzrqp8)

Luckily for me Stefichen5 had already reverse engineered the protocol and had a little rough, but detailed write up about it. The protocol is simple, and consists of a few commands that we can send to the desk to control it. The commands are sent as a series of bytes, and the desk responds with a series of bytes as well. I will try to explain the protocol more in detail in a structured way in two parts: Received Data and Sent Data.

### **Received Data (Desk -> ESP)**

- The desk constantly sends updates about its height.
- Unfortunately, based on my experiments there is some delay in the data, so we can’t rely on it for real-time data. This causes some issues, but we will talk about them later.
- These messages are always 6 bytes long, and they come in two forms:

#### **Message Format:**
| Byte 0 | Byte 1 | Byte 2 | Byte 3 | Byte 4   | Byte 5   |
|--------|--------|--------|--------|----------|----------|
| `0x98` | `0x98` | `0x00` | `0x00` | (height) | (height) |
| `0x98` | `0x98` | `0x03` | `0x03` | (height) | (height) |

- **Bytes 0-1 (`0x98, 0x98`)**: These bytes say, “Hey, I’m the desk.” They never change.
- **Bytes 2-3 (`0x00, 0x00` or `0x03, 0x03`)**: These seem to indicate the desk’s state. Most of the time, it’s `0x00`, but sometimes it’s `0x03`. (We still don’t know why, but it doesn’t seem critical.)
- **Bytes 4-5 (Height)**: This is the desk’s height in a raw format. The two bytes always match, e.g., `0x4B, 0x4B`.

#### **Height Data: What Does It Mean?**

Stefichen5's guide says that the height value (`Bytes 4-5`) ranges from **`0x4B` (lowest)** to **`0x7B` (highest)**. But for my desk these values are incorrect. To find the correct values, I basically moved the desk to the lowest and highest points using the keypad and recorded the values based on the received data. After some contemplation I realized that the recorded and transmitted height data is just the height in centimeters. So the height data is actually the height of the desk in centimeters. Weirdly enough though the keypads screen shows the height in inches.

Note: Stefichen5 gets confused as to why the increments are done in **0.4 inches**, but our finding answers that question. The desk uses a **metric system** for the height data, but the keypad shows the height in inches. So the desk doesn't move in 0.4 inches, but in 1 cm increments.

Each step represents **1 centimenters**, starting at **66 centimeters** (I think I might have the 3-stage, taller model) and going up to **131 centimeters**. So the height data is actually the height of the desk in centimeters.

### **Sent Data (ESP -> Desk)**

To control the desk, we send **5-byte messages**. The format is consistent, making it easy to craft commands:

#### **Message Format**
| Byte 0 | Byte 1 | Byte 2 | Byte 3   | Byte 4   |
|--------|--------|--------|----------|----------|
| `0xD8` | `0xD8` | `0x66` | (button) | (button) |

- **Bytes 0-2 (`0xD8, 0xD8, 0x66`)**: These are the fixed headers. Every command starts this way.
- **Bytes 3-4 (Button)**: These define the action, like moving up, down, or going to a preset. The two bytes are always the same (e.g., `0x02, 0x02` for moving up).

---

#### **Button Actions**
Here’s what each button value does:

| Button Value | Action           |
|--------------|------------------|
| `0x00`       | NOOP (no action) |
| `0x01`       | Move down        |
| `0x02`       | Move up          |
| `0x04`       | Preset 1         |
| `0x08`       | Preset 2         |
| `0x10`       | Preset 3         |
| `0x20`       | Preset 4         |
| `0x40`       | Memory (M) button |

---

Also commands must be sent **every ~50ms** to keep the desk moving. If desk doesn't receive a command in this period, it stops moving. This means we need to keep spamming the desk with commands to mimic a “button held down” effect.

Also the keypad sends a No Operation command at the beginning of every button press. This is to make sure that the desk is awake and ready to receive commands. Another thing is the way we emulate button holding. If we want to say, keep pressing up, we need to send the up command as long as we want the desk to move up. So it would look like this:

```bash
0xD8, 0xD8, 0x66, 0x00, 0x00 # NOOP
0xD8, 0xD8, 0x66, 0x02, 0x02 # Move up
0xD8, 0xD8, 0x66, 0x02, 0x02 # Move up
0xD8, 0xD8, 0x66, 0x02, 0x02 # Move up
...
```

### **Important Discoveries Based on My Testings**

Here’s where our own testing added some important context to **Stefichen5**’s findings:

#### **Waking Up the Desk**
- It seemed like Stefichen5 didn't note down what the red wire does or didn't document it. After a painful discovery period I know what red wire is doing and I know that we need it. This was the major reason I was keep failing in all my attempts.
- The desk won’t respond to commands unless you first “wake it up” by pulling the sleep line (`red wire`) LOW. This is a crucial step that Stefichen5 didn’t mention, but it’s essential for the desk to respond to commands.

#### **Button Commands**
- Every button press needs to be repeated at least twice to take effect. I am unsure why this is the case, but my theory is that even if the desk gets activated by the first command there is simply not enough time for the movement since it doesn't move instantly. So we need to send the command again to make sure the desk moves.

#### **Stops Moving**
- If we want to stop the desk from moving, we need to send double NOOP commands.

#### **Preset Quirks**
- When using presets (`0x04`, `0x08`, etc.), normally we just hold the button on the keypad very little and it just moves even if you leave the button. But with commands, since we need to keep the desk alive by sending commands every 50ms, we need to either keep sending the preset command or NOOP commands to keep the desk moving.

---

### **Protocol Summary**

Here’s a cheat sheet to summarize everything:

#### Desk → Controller (Received Data)
- **Length:** 6 bytes
- **Format:** `0x98, 0x98, 0x00/0x03, 0x00/0x03, (height), (height)`
- **Height Range:** `66 cm` to `131 cm`

#### Controller → Desk (Transmitted Data)
- **Length:** 5 bytes
- **Format:** `0xD8, 0xD8, 0x66, (button), (button)`
- **Button Values:**
  - `0x00`: NOOP (no action)
  - `0x02`: Move up
  - `0x01`: Move down
  - `0x04`: Preset 1
  - `0x08`: Preset 2
  - `0x10`: Preset 3
  - `0x20`: Preset 4
  - `0x40`: Memory (M) button


## Wiring and Initial Setup

Ah, the joys of wires—strip, connect, test, and pray. This part of the project, while intimidating at first, is actually quite straightforward once you know what you're doing (trust me, I had no idea what I was doing at the start). Let’s break it down step by step, from prepping the wires to finalizing everything neatly in an enclosure.

### **Step 1: Understanding the Desk’s Cable**

The Autonomous Desk controller cable (the one connecting the keypad to the desk motor) is your main target here. It has a **10P10C (10-pin modular) connector**, but we don’t need to deal with all ten wires because there is only 5 cables in it (hehe). Here’s the breakdown of the wires:

| Wire Color | Purpose                        |
|------------|--------------------------------|
| Yellow     | Keypad/ESP TX - Desk RX        |
| Green      | Keypad/ESP RX - Desk TX        |
| Red        | Sleep line (Activates the desk)|
| Orange     | Ground (GND)                   |
| Blue       | 5V Input (VIN)                 |

Note: It was a bit hard for me to understand the tables shared by others, because they keep referring to the cables as RX and TX. But in reality a cable is always both RX and TX. It is the perspective that changes. So when you are looking at the desk's cable, the yellow cable is the RX cable, and the green cable is the TX cable. Because RX means the cable that receives data, and TX means the cable that transmits data, and in yellow cable's case, it is receiving data from the desk, and transmitting data to the ESP (or the keypad).

Findind the wires are very easy since the board is very nicely labeled inside the keypad.

![Desk Keypad](./desk_keypad.jpg)

---

### **Step 2: Stripping and Tapping into the Wires**

The easiest way to connect the ESP to the desk is by **tapping into the existing controller cable**. You don’t need to change the connection of the keypad; we’re just “spying” on the signals. Other guides suggest using an external 10P10C connector, but I found that tapping into the existing cable is much simpler and cleaner (and cheaper - 10$ saved yay!).

Here is how you can do it with images (except the first one, because I forgot to take a picture of that):

1. **Cut the Cable**:
   We need to tap into the desk’s controller cable, so we cut it in two pieces to put the ESP in between. You can see the length of both ends in the image, but I think i cut the desk side a bit too short, it was a little annoying while prototyping. You can't really go wrong though, so just cut based on where you wanna place the end product. I wanted it to be close to the desk controller, so I cut desk side short (maybe too short).

![Cutting the Cable](./tapping_into_desk_cable_1.jpg)

2. **Strip the Outer Cable**:
   Carefully remove the outer jacket of both sides to expose the internal wires. I cannot stress enough how good it is to have a universal wire stripper for this. I tried with a knife, with a lighter and with a traditional wire stripper, but when I got the universal one, it was literally a second, and it was perfectly done. I highly recommend getting one. But if you don't have one, you can use a knife or a lighter, but be careful not to cut the inner wires.

![Stripping the Cable](./tapping_into_desk_cable_2.jpg)

3. **Strip the Inner Wires**:
   Once the outer jacket is removed, strip the inner wires to expose the copper conductors.

![Stripping the Wires](./tapping_into_desk_cable_3.jpg)

4. **Tap Into the Wires (Prototyping)**:
   I recommend prototyping and testing first before making the connections permanent. For prototyping, use **Wago connectors** (10 in total) adding them to the end of the exposed wires. This makes it easy to connect and disconnect the ESP during testing. You can also use alligator clips but I found them to be very unreliable, and they kept falling off or touching each other. Off brand wago connectors are super cheap and they are very reliable, so I highly recommend them.

![Prototyping with Wago Connectors](./tapping_into_desk_cable_4.jpg)

5. **Tap Into the Wires (Final Setup)**:
   Once you’ve confirmed that everything works as expected, you can solder the ESP wires directly to the desk cable. This makes the connection permanent and more reliable. I used 20 AWG wires for the connections, but you can use as thin as 26 AWG wires as well. I also used heat shrink tubing to cover the soldered joints and prevent short circuits.

![Final Setup with Soldering](./tapping_into_desk_cable_5.jpg)

---

### **Step 3: Connecting the ESP32**

Now that the desk cable is ready, let’s hook it up to the ESP32. Check the diagram below for the correct connections:

![ESP32 Connections](./esp32_connections.jpg)

And here is the picture of the final setup:

![Final Setup](./final_setup.jpg)

Note: **Diode Placement**: To use both the keypad and the ESP at the same time, we need a diode (e.g., 1N4148) on the **TX line from the keypad**. This prevents the keypad and ESP from conflicting when they both try to send data. The diodes are directional, so make sure the cathode (the side with the stripe) is facing the ESP. Check the images to make sure you are placing the diode correctly.

---

### **Step 4: Prototyping on a Breadboard**

Before soldering everything into place, let’s test the setup using a breadboard. Here’s how to do it:

1. **Plug the ESP into the Breadboard**: Connect the ESP32 to a breadboard for easy wiring. See the jumper cable under the ESP in the image below :) Yeah unfortunately you need that there. Place that first so you don't have to lift the ESP at the second step.

![Breadboard Setup](./breadboard_setup.jpg)

2. **Connect the Wires**: Use jumper cables to connect the ESP pins, desk wires and the keypad using the Wago connectors. Make sure to follow the correct connections.

3. **Test the Connections (Optional)**: Use a multimeter to check for continuity and verify that all connections are secure. You can use the table below for expected voltages:

| Wire/Pin/Connection | Expected Voltage |
|----------------------|------------------|
| ESP TX Pin           | ~3.3V             |
| ESP RX Pin           | ~3.3V             |
| Desk TX Wire         | ~3.3V             |
| Desk RX Wire         | ~3.3V             |
| Red Wire (Sleep)     | ~5V             |

---

### **Step 5: Installing Thonny and Setting Up MicroPython**

To upload the test code to your ESP32, we will use **Thonny**. This is a beginner-friendly Python IDE that supports MicroPython out of the box. Let’s go step by step:

#### **1. Install Thonny**
- Go to the [Thonny website](https://thonny.org/) and download the installer for your operating system (Windows, macOS, or Linux).
- Follow the instructions for your operating system to install Thonny. It's a straightforward process, just click "Next" a few times.
- Open Thonny after installation. It should greet you with a simple, clean interface.

#### **2. Flash MicroPython to the ESP32**
Before running any MicroPython code, you need to install MicroPython firmware on your ESP32:

- Visit the [MicroPython Downloads page](https://micropython.org/download/esp32/) and download the latest stable version of the ESP32 firmware (a `.bin` file).
- Plug your ESP32 into your computer via USB.
- Install esptool by running the following in your terminal:
```bash
pip install esptool
```
- Sometimes, leftover data on the ESP32 can cause issues (Believe me, it does). To erase the flash, run:
```bash
esptool.py --port COMX erase_flash
```
Replace `COMX` with your ESP32’s port (e.g., `/dev/ttyUSB0` on Linux or `COM3` on Windows).
- Run the following command to flash the firmware:
```bash
esptool.py --port COMX --baud 460800 write_flash -z 0x1000 firmware.bin
```
Replace `COMX` with your ESP32’s port and `firmware.bin` with the name of your downloaded MicroPython file.

For example in my case, the command to remove everything and install the micropython was:
```bash
esptool.py --port /dev/tty.SLAB_USBtoUART erase_flash
esptool.py --port /dev/tty.SLAB_USBtoUART --baud 460800 write_flash -z 0x1000 Downloads/ESP32_GENERIC-20241129-v1.24.1.bin
```

Done...

#### **3. Configure Thonny for MicroPython**
1. **Select Interpreter**:
   - In Thonny, go to **Tools > Options > Interpreter**.
   - Set the interpreter to **MicroPython (ESP32)**.
   - Choose the port corresponding to your ESP32. On Windows, it’s usually `COMX`. On macOS/Linux, it’ll look like `/dev/ttyUSB0` or `/dev/ttyS0`. Or in my case `/dev/tty.SLAB_USBtoUART`.

2. **Test the Connection**:
   - Click **Stop/Restart Backend** in Thonny. If successful, you should see a MicroPython REPL (interactive shell) appear at the bottom of the Thonny window.

![Thonny Setup](./thonny_setup.jpg)

---

### **Step 6: Uploading and Testing Code**

With MicroPython flashed and Thonny set up, it’s time to upload and run your test scripts.

1. **Write or Open a Script**:
   - Copy the test scripts into Thonny. Start with something simple, like reading the desk height.

2. **Save the Script to the ESP32**:
   - In Thonny, go to **File > Save As**.
   - Choose **MicroPython device** and save the file as `main.py`. This ensures the script runs automatically when the ESP32 is powered.

3. **Run the Script**:
   - Click the green **Run** button in Thonny to execute your script.

### **Debugging Tips**
- **No Connection?** Check that the correct port is selected in Thonny.
- **No Data?** Double-check your wiring (especially RX/TX connections). This happens so much... Or happened I guess.
- **Weird Characters?** Ensure the baud rate is set to `9600`. But even then you might see weird characters, I think its because of some interference or something. As long as you see the correct data, you are good to go.

Once this test works, you can move on to testing more complex functions like moving the desk or using presets. If you encounter issues, don’t worry—this is where the learning happens. I have written couple isolated test scripts and a large all-functions test script to test the setup. You can use them one by one and finally use the all-functions test script to test everything at once.

## Finalizing the Setup

After testing and confirming that everything works as expected, it’s time to finalize the setup. This involves:
1. Soldering the connections,
2. Uploading the `main.py` script to the ESP32,
3. Placing the ESP in an enclosure, and
4. Mounting it near the desk controller. Well this step is up to you of course.

I am not really going to explain these steps since we already discussed them all in the previous steps. But here are some images/videos following me in different steps:

![Soldering the Connections](./soldering_connections.jpg)

![Soldered Connections](./soldered_connections.jpg)

{{< youtube dyEWGVJLQgc >}} <!-- Soldering time-lapse and final setup -->

![Printed Enclosure](./printed_enclosure.jpg)

{{< youtube dyEWGVJLQgc >}} <!-- Enclosure printing time-lapse -->

![Fitting the ESP in the Enclosure](./fitting_enclosure.jpg)

![Mounting the Enclosure](./mounting_enclosure.jpg)

![Final Setup](./final_setup.jpg)

{{< youtube dyEWGVJLQgc >}} <!-- Final setup video -->

I know I promised to talk about MQTT, and ESP8266 differences, but I think this post is already way too long. I will write a separate post for the MQTT integration and the differences between ESP32 and ESP8266. I hope this post was helpful for you, and I hope you enjoyed reading it. I will see you in the next post. Until then, take care and have fun automating your desk! And do not forget to stand up and stretch every now and then :) (If you enjoyed my work/post consider starring the repo, it keeps me motivated!)