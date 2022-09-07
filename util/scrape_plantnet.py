from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests
import os

save_dir = "/media/data/datasets/plant/classification/ocimum_basilicum"
plantnet_species_url = r'https://identify.plantnet.org/the-plant-list/species/Ocimum%20basilicum%20L./data' 
plant_prefix = "basil"

driver = webdriver.Firefox()
driver.get(plantnet_species_url)
# time.sleep(5)
leaf_el = driver.find_element(By.XPATH, r"/html/body/div[1]/div/div/main/div/div[4]/div/nav/ul/li[2]/a")

driver.execute_script(f'window.scrollTo(0, {leaf_el.location["y"]})')
time.sleep(1)
leaf_el.click()

# Get images under leaves
img_els = driver.find_elements(By.XPATH, r'/html/body/div[1]/div/div/main/div/div[4]/div/div/div[2]/div[1]//img')

# Iterate over all images
for i, img_el in enumerate(img_els):
    # Goto source image
    driver.execute_script(f'window.scrollTo(0, {img_el.location["y"]})')
    
    # Get src image
    src = img_el.get_attribute('src')

    print(f"{i} DOWNLOADING {src} ...")


    while src is None:
        time.sleep(5)
        src = img_el.get_attribute('src')
        print(f"WAITING FOR {src}")

    # Replace small image url with large image url
    src_split = src.split('/s/')
    src = src_split[0] + '/o/' + src_split[1]

    
    # Download the image
    img_data = requests.get(src).content
    with open(os.path.join(save_dir, f'{plant_prefix}_{i}.jpg'), 'wb') as f:
        f.write(img_data)


driver.close()