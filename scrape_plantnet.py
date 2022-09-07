from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import requests

plantnet_species_url = r'https://identify.plantnet.org/useful/species/Lactuca%20sativa%20L./data' 

driver = webdriver.Firefox()
driver.get(plantnet_species_url)
driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
time.sleep(3)
el = driver.find_element(By.XPATH, r"/html/body/div[1]/div/div/main/div/div[4]/div/nav/ul/li[2]/a")
el.click()

# Wait for images to load
time.sleep(10)

driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
time.sleep(20)

# Get images under leafs
img_els = driver.find_elements(By.XPATH, r'/html/body/div[1]/div/div/main/div/div[4]/div/div/div[2]/div[1]//img')


# time.sleep(10)

# Iterate over all images
for i, img_el in enumerate(img_els):


    if i <= 100:
        continue

    # Get src image
    src = img_el.get_attribute('src')
    print(f"{i} DOWNLOADING {src} ...")


    if src is None:
        break
    # Replace small image url with large image url
    src_split = src.split('/s/')
    src = src_split[0] + '/o/' + src_split[1]

    # Download the image
    img_data = requests.get(src).content
    with open(f'test_dataset/lettuce_{i}.jpg', 'wb') as f:
        f.write(img_data)


driver.close()