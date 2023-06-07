import os
import re
import subprocess
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor

output_path = '/efs/hang/telecombrain/globecom23/ALL3GPP_RAW_DATA'

working_groups = [
    #{'name': 'CT1', 'abbr': 'C1-', 'url': 'https://www.3gpp.org/ftp/tsg_ct/WG1_mm-cc-sm_ex-CN1/'},
    #{'name': 'CT3', 'abbr': 'C3-', 'url': 'https://www.3gpp.org/ftp/tsg_ct/WG3_interworking_ex-CN3/'},
    #{'name': 'CT4', 'abbr': 'C4-', 'url': 'https://www.3gpp.org/ftp/tsg_ct/WG4_protocollars_ex-CN4/'},
    #{'name': 'CT6', 'abbr': 'C6-', 'url': 'https://www.3gpp.org/ftp/tsg_ct/WG6_Smartcard_Ex-T3'},

    #{'name': 'RAN1', 'abbr': 'R1-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG1_RL1/'},
    #{'name': 'RAN2', 'abbr': 'R2-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG2_RL2'},
    #{'name': 'RAN3', 'abbr': 'R3-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG3_Iu/'},
    #{'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio'},
    #{'name': 'RAN5', 'abbr': 'R5-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG5_Test_ex-T1'},
    #{'name': 'RAN AH1', 'abbr': 'RT-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/AHG1_ITU_Coord/TSGRT_ALL'},
    #{'name': 'SA1', 'abbr': 'S1-', 'url': 'https://www.3gpp.org/ftp/tsg_sa/WG1_Serv/'},
    #{'name': 'SA2', 'abbr': 'S2-', 'url': 'https://www.3gpp.org/ftp/tsg_sa/WG2_Arch/'},
    #{'name': 'SA3', 'abbr': 'S3-', 'url': 'https://www.3gpp.org/ftp/tsg_sa/WG3_Security'},
    #{'name': 'SA4', 'abbr': 'S4-', 'url': 'https://www.3gpp.org/ftp/tsg_sa/WG4_CODEC'},
    #{'name': 'SA5', 'abbr': 'S5-', 'url': 'https://www.3gpp.org/ftp/tsg_sa/WG5_TM'},
    #{'name': 'SA6', 'abbr': 'S6-', 'url': 'https://www.3gpp.org/ftp/tsg_sa/WG6_MissionCritical'}

   # unfinished!!!!
   {'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_93/Docs'},
   {'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_92Bis/Docs'}, 
   {'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_94/Docs'},
   {'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_94Bis/Docs'},
   {'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_94_e/Docs'},
   {'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_94_eBis/Docs'},
   {'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_95/Docs'},
   {'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_95_e/Docs'}, 
   {'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_96_e/Docs'},
   {'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_97_e/Docs'},
    ##{'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_97bis'}, 
   {'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_98_e/Docs'},
   {'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_98bis_e/Docs'},
   {'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_99-e/Docs'},
    ##{'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_99bis/Docs'}, 
   {'name': 'RAN4', 'abbr': 'R4-', 'url': 'https://www.3gpp.org/ftp/tsg_ran/WG4_Radio/TSGR4_100-e/Docs'},

]


def download_files(url, wg_abbr, wg_name, target_folder, depth=0):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    if not table:
        return

    links = table.find_all('a')

    with ThreadPoolExecutor(max_workers=10) as executor:
        for link in links:
            href = link.get('href')
            if not href or not is_valid_url(href):
                continue

            href_last_part = os.path.basename(href)

            # general condition
            #if depth == 0 and not href_last_part.startswith('TSG'):

            # condition for rest of CT1
            #if depth == 0 and not ('TSGC1_128e' <= href_last_part <= 'TSGC1_145_Chicago'):
            # condition for rest of CT6
            #if depth == 0 and not href_last_part.startswith('CT'):
            # condition for rest of  RAN1
            #if depth == 0 and not ('TSGR1_109-e' <= href_last_part <= 'TSGR1_109-e'):
                #continue

            #if re.match(rf'^[{wg_abbr.upper()}{wg_abbr.lower()}].*\.zip$', href_last_part):  # File
            if href_last_part.endswith('.zip'):  # File
                file_url = urljoin(url, href)
                local_file_path = os.path.join(target_folder, href_last_part)
                executor.submit(download_file, file_url, local_file_path, wg_name)
            elif not href_last_part.endswith('.zip'):  # Directory
                next_url = urljoin(url, href)
                executor.submit(download_files, next_url, wg_abbr, wg_name, target_folder, depth + 1)


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def download_file(url, file_path, wg_name):
    try:
        subprocess.run(['wget', '-q', '-O', file_path, url], check=True)
        print(f"{os.path.basename(file_path)} of working group {wg_name} is successfully downloaded!\n")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file {os.path.basename(file_path)}: {str(e)}\n")

#for wg in working_groups:
    #wg_name = wg['name']
    #wg_url = wg['url']
    #wg_abbr = wg['abbr']
    #target_folder = os.path.join(output_path, wg_name)
    #os.makedirs(target_folder, exist_ok=True)
    #download_files(wg_url, wg_abbr, wg_name, target_folder)


#base_output_path = '/efs/hang/telecombrain/globecom23/ALL3GPP_RAW_DATA'

for wg in working_groups:
    print(f"Downloading files for working group {wg['name']}...")
    target_folder = os.path.join(output_path, wg['name'])
    os.makedirs(target_folder, exist_ok=True)
    # Download files for the current working group
    download_files(wg['url'], wg['abbr'], wg['name'], target_folder)
    print(f"Finished downloading remaining files for working group {wg['name']}!!!\n\n")

