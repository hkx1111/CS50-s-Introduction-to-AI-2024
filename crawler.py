import os
import requests
import time
from bs4 import BeautifulSoup

# Base URL for the CS50 AI course
base_url = "https://cs50.harvard.edu/ai/2024/notes/"

# Function to create directories and save content
def save_lecture_notes(lecture_num, content):
    # Create directory if it doesn't exist
    directory = f"Lecture{lecture_num}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save content to file
    with open(f"{directory}/notes.txt", "w", encoding="utf-8") as file:
        file.write(content)
    
    print(f"Saved notes for Lecture {lecture_num}")

# Function to extract text with basic formatting
def extract_formatted_text(element):
    result = ""
    
    # Process all elements in the main content
    for elem in element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'pre', 'code']):
        # Handle headings
        if elem.name.startswith('h'):
            level = int(elem.name[1])
            result += '\n' + '#' * level + ' ' + elem.get_text().strip() + '\n\n'
        
        # Handle paragraphs
        elif elem.name == 'p':
            result += elem.get_text().strip() + '\n\n'
        
        # Handle lists
        elif elem.name == 'ul' or elem.name == 'ol':
            for li in elem.find_all('li'):
                result += '- ' + li.get_text().strip() + '\n'
            result += '\n'
        
        # Handle code blocks
        elif elem.name == 'pre':
            result += '\n' + elem.get_text() + '\n\n'
        
        # Handle inline code (if not within a pre)
        elif elem.name == 'code' and elem.parent.name != 'pre':
            result += elem.get_text() + ' '
    
    return result

# Main function to scrape lectures
def scrape_lectures():
    for lecture_num in range(1, 7):  # Lectures 1-6
        # Create URL for current lecture
        url = f"{base_url}{lecture_num}/"
        
        try:
            print(f"Downloading Lecture {lecture_num} from {url}")
            
            # Download the webpage
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract the main content
            main_content = soup.find("main")
            
            if main_content:
                # Extract formatted text
                formatted_text = extract_formatted_text(main_content)
                
                # Add source information
                full_text = f"CS50 AI - Lecture {lecture_num} Notes\n"
                full_text += f"Source: {url}\n\n"
                full_text += formatted_text
                
                # Save the lecture notes
                save_lecture_notes(lecture_num, full_text)
            else:
                print(f"Could not find main content for Lecture {lecture_num}")
            
            # Be nice to the server with a delay between requests
            time.sleep(2)
                
        except Exception as e:
            print(f"Error scraping Lecture {lecture_num}: {e}")

# Run the scraper
if __name__ == "__main__":
    print("Starting to scrape CS50 AI lecture notes...")
    scrape_lectures()
    print("Finished scraping all lectures.")
