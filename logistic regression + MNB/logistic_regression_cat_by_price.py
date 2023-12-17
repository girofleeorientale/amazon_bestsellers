import pandas as pd
import numpy as np

# import the class
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16, max_iter=200)
logreg_1 = LogisticRegression(random_state=16)

first = True

df_y_test_total = []
data_y_pred = []

mean_ = 385
rev_stdev = 3827
mean_bought = 18
stdev_bought = 158

for chunk in pd.read_csv('train.csv', chunksize = 100000):

    chunk['reviewNormalized'] = (chunk['reviews'] - mean_)/rev_stdev
    #print(chunk[['reviews', 'reviewNormalized']].head())

    # create column for 1 if price under 15, 2: 15-30 etc
    chunk['categoryMedianPrice'] = 0
    chunk.loc[(chunk['categoryName'] == 'Boys') | (chunk['categoryName'] == 'Handmade Home & Kitchen Products'
                                                        ) | (chunk['categoryName'] == 'Car & Motorbike'
                                                        ) | (chunk['categoryName'] == 'Hardware'
                                                        ) | (chunk['categoryName'] == 'Wearable Technology'
                                                        ) | (chunk['categoryName'] == 'USB Gadgets'
                                                        ) | (chunk['categoryName'] == 'Light Bulbs'
                                                        ) | (chunk['categoryName'] == 'Handmade Gifts'
                                                        ) | (chunk['categoryName'] == 'Calendars & Personal Organisers'
                                                        ) | (chunk['categoryName'] == 'Pet Supplies'
                                                        ) | (chunk['categoryName'] == 'Plants, Seeds & Bulbs'
                                                        ) | (chunk['categoryName'] == 'Kids\' Art & Craft Supplies'
                                                        ) | (chunk['categoryName'] == 'Skin Care'
                                                        ) | (chunk['categoryName'] == 'Hobbies'
                                                        ) | (chunk['categoryName'] == 'Computer Screws'
                                                        ) | (chunk['categoryName'] == 'Bath & Body'
                                                        ) | (chunk['categoryName'] == 'Handmade Kitchen & Dining'
                                                        ) | (chunk['categoryName'] == 'Agricultural Equipment & Supplies'
                                                        ) | (chunk['categoryName'] == 'Power & Hand Tools'
                                                        ) | (chunk['categoryName'] == 'Boating Footwear'
                                                        ) | (chunk['categoryName'] == 'SIM Cards'
                                                        ) | (chunk['categoryName'] == 'Cutting Tools'
                                                        ) | (chunk['categoryName'] == 'Abrasive & Finishing Products'
                                                        ) | (chunk['categoryName'] == 'Mobile Phone Accessories'
                                                        ) | (chunk['categoryName'] == 'Cables & Accessories'
                                                        ) | (chunk['categoryName'] == 'Manicure & Pedicure Products'
                                                        ) | (chunk['categoryName'] == 'Rough Plumbing'
                                                        ) | (chunk['categoryName'] == 'Kitchen Tools & Gadgets'
                                                        ) | (chunk['categoryName'] == 'Cushions & Accessories'
                                                        ) | (chunk['categoryName'] == 'Pens, Pencils & Writing Supplies'
                                                        ) | (chunk['categoryName'] == 'School & Educational Supplies'
                                                        ) | (chunk['categoryName'] == 'Home Fragrance'
                                                        ) | (chunk['categoryName'] == 'Games & Game Accessories'
                                                        ) | (chunk['categoryName'] == 'Beauty'
                                                        ) | (chunk['categoryName'] == 'Bedding Accessories'
                                                        ) | (chunk['categoryName'] == 'Kitchen Linen'
                                                        ) | (chunk['categoryName'] == 'Hiking Games & Game Accessories'
                                                        ) | (chunk['categoryName'] == 'Handmade Artwork'
                                                        ) | (chunk['categoryName'] == 'Make-up'
                                                        ) | (chunk['categoryName'] == 'Industrial Electrical'
                                                        ) | (chunk['categoryName'] == 'Painting Supplies, Tools & Wall Treatments'
                                                        ) | (chunk['categoryName'] == 'Electrical Power Accessories'
                                                        ) | (chunk['categoryName'] == 'Safety & Security'
                                                        ) | (chunk['categoryName'] == 'Vacuums & Floorcare'
                                                        ) | (chunk['categoryName'] == 'Radio Communication'
                                                        ) | (chunk['categoryName'] == 'Girls'
                                                        ) | (chunk['categoryName'] == 'Tablet Accessories'
                                                        ) | (chunk['categoryName'] == 'Baby'
                                                        ) | (chunk['categoryName'] == 'Office Supplies'
                                                        ) | (chunk['categoryName'] == 'Baby & Toddler Toys'
                                                        ) | (chunk['categoryName'] == 'Learning & Education Toys'
                                                        ) | (chunk['categoryName'] == 'Health & Personal Care'
                                                        ) | (chunk['categoryName'] == 'Signs & Plaques'
                                                        ) | (chunk['categoryName'] == 'Gardening'
                                                        ) | (chunk['categoryName'] == 'Decorative Artificial Flora'
                                                        ) | (chunk['categoryName'] == 'Toy Advent Calendars'
                                                        ) | (chunk['categoryName'] == 'Candles & Holders'
                                                        ) | (chunk['categoryName'] == 'External Sound Cards'
                                                        ) | (chunk['categoryName'] == 'Handmade Clothing, Shoes & Accessories'
                                                        ) | (chunk['categoryName'] == 'Handmade Home Décor'
                                                        ) | (chunk['categoryName'] == 'Professional Education Supplies'
                                                        ) | (chunk['categoryName'] == 'Handmade'
                                                        ) | (chunk['categoryName'] == 'Headphones, Earphones & Accessories'
                                                        ) | (chunk['categoryName'] == 'Ironing & Steamers'
                                                        ) | (chunk['categoryName'] == 'Household Batteries, Chargers & Accessories'
                                                        ) | (chunk['categoryName'] == 'Grocery'
                                                        ) | (chunk['categoryName'] == 'Hi-Fi & Home Audio Accessories'
                                                        ) | (chunk['categoryName'] == 'Bakeware'
                                                        ) | (chunk['categoryName'] == 'Torches'
                                                        ) | (chunk['categoryName'] == 'Electrical'
                                                        ) | (chunk['categoryName'] == 'Adapters'
                                                        ) | (chunk['categoryName'] == 'Computer Memory Card Accessories'
                                                        ) | (chunk['categoryName'] == 'Hard Drive Accessories'
                                                        ) | (chunk['categoryName'] == 'Garden Tools & Watering Equipment'
                                                        ) | (chunk['categoryName'] == 'Office Paper'
                                                        ) | (chunk['categoryName'] == 'Jigsaws & Puzzles'
                                                        ) | (chunk['categoryName'] == 'Decorative Home Accessories'
                                                        ) | (chunk['categoryName'] == 'Clocks'
                                                        ) | (chunk['categoryName'] == 'Doormats'
                                                        ) | (chunk['categoryName'] == 'Photo Frames'
                                                        ) | (chunk['categoryName'] == 'Kids\' Dress Up & Pretend Play'
                                                        ) | (chunk['categoryName'] == 'Soft Toys'
                                                        ) | (chunk['categoryName'] == 'Handmade Jewellery'
                                                        ) | (chunk['categoryName'] == 'Hair Care'
                                                        ) | (chunk['categoryName'] == 'Arts & Crafts'
                                                        ) , 'categoryMedianPrice'] = 1
    
    # 15 to 30
    chunk.loc[(chunk['categoryName'] == 'Film Cameras') | (chunk['categoryName'] == 'PC Gaming Accessories'
                                                        ) | (chunk['categoryName'] == 'Microphones'
                                                        ) | (chunk['categoryName'] == 'Sports & Outdoors'
                                                        ) | (chunk['categoryName'] == 'Heating, Cooling & Air Quality'
                                                        ) | (chunk['categoryName'] == 'Storage & Home Organisation'
                                                        ) | (chunk['categoryName'] == 'Motorbike Accessories'
                                                        ) | (chunk['categoryName'] == 'Motorbike Chassis'
                                                        ) | (chunk['categoryName'] == 'Bathroom Lighting'
                                                        ) | (chunk['categoryName'] == 'Blank Media Cases & Wallets'
                                                        ) | (chunk['categoryName'] == 'Motorbike Lighting'
                                                        ) | (chunk['categoryName'] == 'Indoor Lighting'
                                                        ) | (chunk['categoryName'] == 'Fragrances'
                                                        ) | (chunk['categoryName'] == 'General Music-Making Accessories'
                                                        ) | (chunk['categoryName'] == 'Storage & Organisation'
                                                        ) | (chunk['categoryName'] == 'Kids\' Play Vehicles'
                                                        ) | (chunk['categoryName'] == 'Ski Goggles'
                                                        ) | (chunk['categoryName'] == 'Curtain & Blind Accessories'
                                                        ) | (chunk['categoryName'] == 'Bird & Wildlife Care'
                                                        ) | (chunk['categoryName'] == 'Bedding & Linen'
                                                        ) | (chunk['categoryName'] == 'Birthday Gifts'
                                                        ) | (chunk['categoryName'] == 'PC & Video Games'
                                                        ) | (chunk['categoryName'] == 'Luxury Food & Drink'
                                                        ) | (chunk['categoryName'] == 'Mowers & Outdoor Power Tools'
                                                        ) | (chunk['categoryName'] == 'Kitchen & Bath Fixtures'
                                                        ) | (chunk['categoryName'] == 'Tableware'
                                                        ) | (chunk['categoryName'] == 'Beer, Wine & Spirits'
                                                        ) | (chunk['categoryName'] == 'Kitchen Storage & Organisation'
                                                        ) | (chunk['categoryName'] == 'Water Coolers, Filters & Cartridges'
                                                        ) | (chunk['categoryName'] == 'Car & Vehicle Electronics'
                                                        ) | (chunk['categoryName'] == 'Motorbike Engines & Engine Parts'
                                                        ) | (chunk['categoryName'] == 'Home Brewing & Wine Making'
                                                        ) | (chunk['categoryName'] == 'Computers, Components & Accessories'
                                                        ) | (chunk['categoryName'] == 'Motorbike Brakes'
                                                        ) | (chunk['categoryName'] == 'Lighting'
                                                        ) | (chunk['categoryName'] == 'eBook Readers & Accessories'
                                                        ) | (chunk['categoryName'] == 'Small Kitchen Appliances'
                                                        ) | (chunk['categoryName'] == 'Mattress Pads & Toppers'
                                                        ) | (chunk['categoryName'] == 'Children\'s Bedding'
                                                        ) | (chunk['categoryName'] == 'Window Treatments'
                                                        ) | (chunk['categoryName'] == 'Dolls & Accessories'
                                                        ) | (chunk['categoryName'] == 'Printers & Accessories'
                                                        ) | (chunk['categoryName'] == 'Sports Toys & Outdoor'
                                                        ) | (chunk['categoryName'] == 'I/O Port Cards'
                                                        ) | (chunk['categoryName'] == 'Telephones, VoIP & Accessories'
                                                        ) | (chunk['categoryName'] == '3D Printing & Scanning'
                                                        ) | (chunk['categoryName'] == 'Outdoor Rope Lights'
                                                        ) | (chunk['categoryName'] == 'Keyboards, Mice & Input Devices'
                                                        ) | (chunk['categoryName'] == 'Mobile Phones & Communication'
                                                        ) | (chunk['categoryName'] == 'Lights and switches'
                                                        ) | (chunk['categoryName'] == 'Photo Printers'
                                                        ) | (chunk['categoryName'] == 'Plugs'
                                                        ) | (chunk['categoryName'] == 'Office Electronics'
                                                        ) | (chunk['categoryName'] == 'Piano & Keyboard'
                                                        ) | (chunk['categoryName'] == 'Synthesisers, Samplers & Digital Instruments'
                                                        ) | (chunk['categoryName'] == 'Headphones & Earphones'
                                                        ) | (chunk['categoryName'] == 'Laptop Accessories'
                                                        ) | (chunk['categoryName'] == 'Drums & Percussion'
                                                        ) | (chunk['categoryName'] == 'Piano & Keyboard'
                                                        ) | (chunk['categoryName'] == 'Pools, Hot Tubs & Supplies'
                                                        ) | (chunk['categoryName'] == 'Outdoor Cooking'
                                                        ) | (chunk['categoryName'] == 'Kids\' Play Figures'
                                                        ) | (chunk['categoryName'] == 'Thermometers & Meteorological Instruments'
                                                        ) | (chunk['categoryName'] == 'Bathroom Furniture'
                                                        ) | (chunk['categoryName'] == 'Bathroom Linen'
                                                        ) | (chunk['categoryName'] == 'Electronic Toys'
                                                        ) | (chunk['categoryName'] == 'External TV Tuners & Video Capture Cards'
                                                        ) | (chunk['categoryName'] == 'Table Tennis'
                                                        ) | (chunk['categoryName'] == 'Billiard, Snooker & Pool'
                                                        ) | (chunk['categoryName'] == 'Bowling'
                                                        ) | (chunk['categoryName'] == 'Trampolines & Accessories'
                                                        ) | (chunk['categoryName'] == 'Sports Supplements'
                                                        ) | (chunk['categoryName'] == 'Camera & Photo Accessories'
                                                        ) | (chunk['categoryName'] == 'Men'
                                                        ) | (chunk['categoryName'] == 'Customers\' Most Loved'
                                                        ) | (chunk['categoryName'] == 'Ballet & Dancing Footwear'
                                                        ) | (chunk['categoryName'] == 'Home Cinema, TV & Video'
                                                        ) | (chunk['categoryName'] == 'Portable Sound & Video Products'
                                                        ) | (chunk['categoryName'] == 'Outdoor Lighting'
                                                        ) | (chunk['categoryName'] == 'Luggage and travel gear'
                                                        ) | (chunk['categoryName'] == 'DJ & VJ Equipment'
                                                        ) | (chunk['categoryName'] == 'Handmade Baby Products'
                                                        ) | (chunk['categoryName'] == 'Construction Machinery'
                                                        ) | (chunk['categoryName'] == 'USB Hubs'
                                                        ) | (chunk['categoryName'] == 'Printer Accessories'
                                                        ) | (chunk['categoryName'] == 'Rugs, Pads & Protectors'
                                                        ) | (chunk['categoryName'] == 'Building & Construction Toys,'
                                                        ) | (chunk['categoryName'] == 'Hallway Furniture'
                                                        ) | (chunk['categoryName'] == 'Networking Devices'
                                                        ) | (chunk['categoryName'] == 'Snow Sledding Equipment'
                                                        ) | (chunk['categoryName'] == 'Vases'
                                                        ) | (chunk['categoryName'] == 'Mirrors'
                                                        ) | (chunk['categoryName'] == 'Slipcovers'
                                                        ) | (chunk['categoryName'] == 'Boxes & Organisers'
                                                        ) | (chunk['categoryName'] == 'Remote & App-Controlled'
                                                        ) | (chunk['categoryName'] == 'Data Storage'
                                                        ) | (chunk['categoryName'] == 'External Optical Drives'
                                                        ) | (chunk['categoryName'] == 'Network Cards'
                                                        ) | (chunk['categoryName'] == 'Women'
                                                        ) | (chunk['categoryName'] == 'Gifts for Her'
                                                        ) | (chunk['categoryName'] == 'Gifts for Him'
                                                        ) | (chunk['categoryName'] == 'Hi-Fi Receivers & Separates'
                                                        ) | (chunk['categoryName'] == 'Coffee, Tea & Espresso'
                                                        ) | (chunk['categoryName'] == 'GPS, Finders & Accessories'
                                                        ) | (chunk['categoryName'] == 'Climbing Footwear'
                                                        ) | (chunk['categoryName'] == 'Equestrian Sports Boots'
                                                        ) , 'categoryMedianPrice'] = 0

    # 30 to 50
    chunk.loc[(chunk['categoryName'] == 'Internal Optical Drives') | (chunk['categoryName'] == 'CD'
                                                        ) | (chunk['categoryName'] == 'Disc & Tape Players'
                                                        ) | (chunk['categoryName'] == 'Motorbike Clothing'
                                                        ) | (chunk['categoryName'] == 'Fireplaces, Stoves & Accessories'
                                                        ) | (chunk['categoryName'] == 'Hi-Fi Speakers'
                                                        ) | (chunk['categoryName'] == 'Smart Speakers'
                                                        ) | (chunk['categoryName'] == 'Furniture & Lighting'
                                                        ) | (chunk['categoryName'] == 'Karaoke Equipment'
                                                        ) | (chunk['categoryName'] == 'String Instruments'
                                                        ) | (chunk['categoryName'] == 'Garden Furniture & Accessories'
                                                        ) | (chunk['categoryName'] == 'Living Room Furniture'
                                                        ) | (chunk['categoryName'] == 'Computer Memory'
                                                        ) | (chunk['categoryName'] == 'Tennis Shoes'
                                                        ) | (chunk['categoryName'] == 'Packaging & Shipping Supplies'
                                                        ) | (chunk['categoryName'] == 'Professional Medical Supplies'
                                                        ) | (chunk['categoryName'] == 'Cookware'
                                                        ) | (chunk['categoryName'] == 'Monitor Accessories'
                                                        ) | (chunk['categoryName'] == 'Radios & Boomboxes'
                                                        ) | (chunk['categoryName'] == 'Home Entertainment Furniture'
                                                        ) | (chunk['categoryName'] == 'Water Ski Clothing'
                                                        ) | (chunk['categoryName'] == 'Ski Clothing'
                                                        ) | (chunk['categoryName'] == 'KVM Switches'
                                                        ) | (chunk['categoryName'] == 'Tripods & Monopods'
                                                        ) | (chunk['categoryName'] == 'Motorbike Electrical & Batteries'
                                                        ) | (chunk['categoryName'] == 'Surveillance Cameras'
                                                        ) | (chunk['categoryName'] == 'Motorbike Instruments'
                                                        ) | (chunk['categoryName'] == 'Darts & Dartboards'
                                                        ) | (chunk['categoryName'] == 'Cricket Shoes'
                                                        ) | (chunk['categoryName'] == 'Computer Audio & Video Accessories'
                                                        ) | (chunk['categoryName'] == 'Recording & Computer'
                                                        ) | (chunk['categoryName'] == 'Downhill Skis'
                                                        ) , 'categoryMedianPrice'] = 0
    
    # 50 to 75
    chunk.loc[(chunk['categoryName'] == 'Motorbike Batteries') | (chunk['categoryName'] == 'Hiking Hand & Foot Warmers'
                                                        ) | (chunk['categoryName'] == 'Women\'s Sports & Outdoor Shoes'
                                                        ) | (chunk['categoryName'] == 'Power Supplies'
                                                        ) | (chunk['categoryName'] == 'Streaming Clients'
                                                        ) | (chunk['categoryName'] == 'Cycling Shoes'
                                                        ) | (chunk['categoryName'] == 'Motorbike Drive & Gears'
                                                        ) | (chunk['categoryName'] == 'Flashes'
                                                        ) | (chunk['categoryName'] == 'Motorbike Handlebars, Controls & Grips'
                                                        ) | (chunk['categoryName'] == 'Made in Italy Handmade'
                                                        ) | (chunk['categoryName'] == 'Computer Cases'
                                                        ) | (chunk['categoryName'] == 'Basketball Footwear'
                                                        ) | (chunk['categoryName'] == 'Smartwatches'
                                                        ) | (chunk['categoryName'] == 'Inflatable Beds, Pillows & Accessories'
                                                        ) | (chunk['categoryName'] == 'Scanners & Accessories'
                                                        ) | (chunk['categoryName'] == 'Internal TV Tuner & Video Capture Cards'
                                                        ) | (chunk['categoryName'] == 'Boxing Shoes'
                                                        ) | (chunk['categoryName'] == 'Camcorders'
                                                        ) | (chunk['categoryName'] == 'Golf Shoes'
                                                        ) | (chunk['categoryName'] == 'Hydraulics, Pneumatics & Plumbing'
                                                        ) | (chunk['categoryName'] == 'Uninterruptible Power Supply Units & Accessories'
                                                        ) | (chunk['categoryName'] == 'Bass Guitars & Gear'
                                                        ) | (chunk['categoryName'] == 'Media Streaming Devices'
                                                        ) | (chunk['categoryName'] == 'Hockey Shoes'
                                                        ), 'categoryMedianPrice'] = 0

    feature_cols = ['reviewNormalized', 'boughtInLastMonth', 'price', 'categoryMedianPrice']
    y = (chunk['isBestSeller'])
    X = (chunk[feature_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

    df_y_test_total.append(y_test)

    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    if first:
        logreg_1.fit(X_train, y_train)
        first = False

    data_y_pred.append(y_pred)
    


final_y_test = pd.concat(df_y_test_total)
arr = np.concatenate(data_y_pred).ravel()

cm = confusion_matrix(final_y_test, arr, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=logreg.classes_)
disp.plot()
plt.show()


df_test = pd.read_csv('test.csv')

df_test_intermed = df_test[['reviews', 'boughtInLastMonth', 'price', 'categoryName', 'isBestSeller']].copy()
df_test_intermed['reviewNormalized'] = (df_test_intermed['reviews'] - mean_)/rev_stdev


df_test_intermed['categoryMedianPrice'] = 0
chunk.loc[(df_test_intermed['categoryName'] == 'Boys') | (chunk['categoryName'] == 'Handmade Home & Kitchen Products'
                                                        ) | (chunk['categoryName'] == 'Car & Motorbike'
                                                        ) | (chunk['categoryName'] == 'Hardware'
                                                        ) | (chunk['categoryName'] == 'Wearable Technology'
                                                        ) | (chunk['categoryName'] == 'USB Gadgets'
                                                        ) | (chunk['categoryName'] == 'Light Bulbs'
                                                        ) | (chunk['categoryName'] == 'Handmade Gifts'
                                                        ) | (chunk['categoryName'] == 'Calendars & Personal Organisers'
                                                        ) | (chunk['categoryName'] == 'Pet Supplies'
                                                        ) | (chunk['categoryName'] == 'Plants, Seeds & Bulbs'
                                                        ) | (chunk['categoryName'] == 'Kids\' Art & Craft Supplies'
                                                        ) | (chunk['categoryName'] == 'Skin Care'
                                                        ) | (chunk['categoryName'] == 'Hobbies'
                                                        ) | (chunk['categoryName'] == 'Computer Screws'
                                                        ) | (chunk['categoryName'] == 'Bath & Body'
                                                        ) | (chunk['categoryName'] == 'Handmade Kitchen & Dining'
                                                        ) | (chunk['categoryName'] == 'Agricultural Equipment & Supplies'
                                                        ) | (chunk['categoryName'] == 'Power & Hand Tools'
                                                        ) | (chunk['categoryName'] == 'Boating Footwear'
                                                        ) | (chunk['categoryName'] == 'SIM Cards'
                                                        ) | (chunk['categoryName'] == 'Cutting Tools'
                                                        ) | (chunk['categoryName'] == 'Abrasive & Finishing Products'
                                                        ) | (chunk['categoryName'] == 'Mobile Phone Accessories'
                                                        ) | (chunk['categoryName'] == 'Cables & Accessories'
                                                        ) | (chunk['categoryName'] == 'Manicure & Pedicure Products'
                                                        ) | (chunk['categoryName'] == 'Rough Plumbing'
                                                        ) | (chunk['categoryName'] == 'Kitchen Tools & Gadgets'
                                                        ) | (chunk['categoryName'] == 'Cushions & Accessories'
                                                        ) | (chunk['categoryName'] == 'Pens, Pencils & Writing Supplies'
                                                        ) | (chunk['categoryName'] == 'School & Educational Supplies'
                                                        ) | (chunk['categoryName'] == 'Home Fragrance'
                                                        ) | (chunk['categoryName'] == 'Games & Game Accessories'
                                                        ) | (chunk['categoryName'] == 'Beauty'
                                                        ) | (chunk['categoryName'] == 'Bedding Accessories'
                                                        ) | (chunk['categoryName'] == 'Kitchen Linen'
                                                        ) | (chunk['categoryName'] == 'Hiking Games & Game Accessories'
                                                        ) | (chunk['categoryName'] == 'Handmade Artwork'
                                                        ) | (chunk['categoryName'] == 'Make-up'
                                                        ) | (chunk['categoryName'] == 'Industrial Electrical'
                                                        ) | (chunk['categoryName'] == 'Painting Supplies, Tools & Wall Treatments'
                                                        ) | (chunk['categoryName'] == 'Electrical Power Accessories'
                                                        ) | (chunk['categoryName'] == 'Safety & Security'
                                                        ) | (chunk['categoryName'] == 'Vacuums & Floorcare'
                                                        ) | (chunk['categoryName'] == 'Radio Communication'
                                                        ) | (chunk['categoryName'] == 'Girls'
                                                        ) | (chunk['categoryName'] == 'Tablet Accessories'
                                                        ) | (chunk['categoryName'] == 'Baby'
                                                        ) | (chunk['categoryName'] == 'Office Supplies'
                                                        ) | (chunk['categoryName'] == 'Baby & Toddler Toys'
                                                        ) | (chunk['categoryName'] == 'Learning & Education Toys'
                                                        ) | (chunk['categoryName'] == 'Health & Personal Care'
                                                        ) | (chunk['categoryName'] == 'Signs & Plaques'
                                                        ) | (chunk['categoryName'] == 'Gardening'
                                                        ) | (chunk['categoryName'] == 'Decorative Artificial Flora'
                                                        ) | (chunk['categoryName'] == 'Toy Advent Calendars'
                                                        ) | (chunk['categoryName'] == 'Candles & Holders'
                                                        ) | (chunk['categoryName'] == 'External Sound Cards'
                                                        ) | (chunk['categoryName'] == 'Handmade Clothing, Shoes & Accessories'
                                                        ) | (chunk['categoryName'] == 'Handmade Home Décor'
                                                        ) | (chunk['categoryName'] == 'Professional Education Supplies'
                                                        ) | (chunk['categoryName'] == 'Handmade'
                                                        ) | (chunk['categoryName'] == 'Headphones, Earphones & Accessories'
                                                        ) | (chunk['categoryName'] == 'Ironing & Steamers'
                                                        ) | (chunk['categoryName'] == 'Household Batteries, Chargers & Accessories'
                                                        ) | (chunk['categoryName'] == 'Grocery'
                                                        ) | (chunk['categoryName'] == 'Hi-Fi & Home Audio Accessories'
                                                        ) | (chunk['categoryName'] == 'Bakeware'
                                                        ) | (chunk['categoryName'] == 'Torches'
                                                        ) | (chunk['categoryName'] == 'Electrical'
                                                        ) | (chunk['categoryName'] == 'Adapters'
                                                        ) | (chunk['categoryName'] == 'Computer Memory Card Accessories'
                                                        ) | (chunk['categoryName'] == 'Hard Drive Accessories'
                                                        ) | (chunk['categoryName'] == 'Garden Tools & Watering Equipment'
                                                        ) | (chunk['categoryName'] == 'Office Paper'
                                                        ) | (chunk['categoryName'] == 'Jigsaws & Puzzles'
                                                        ) | (chunk['categoryName'] == 'Decorative Home Accessories'
                                                        ) | (chunk['categoryName'] == 'Clocks'
                                                        ) | (chunk['categoryName'] == 'Doormats'
                                                        ) | (chunk['categoryName'] == 'Photo Frames'
                                                        ) | (chunk['categoryName'] == 'Kids\' Dress Up & Pretend Play'
                                                        ) | (chunk['categoryName'] == 'Soft Toys'
                                                        ) | (chunk['categoryName'] == 'Handmade Jewellery'
                                                        ) | (chunk['categoryName'] == 'Hair Care'
                                                        ) | (chunk['categoryName'] == 'Arts & Crafts'
                                                        ) , 'categoryMedianPrice'] = 1

feature_cols = ['reviewNormalized', 'boughtInLastMonth', 'price', 'categoryMedianPrice']



y_test = (df_test_intermed['isBestSeller'])
X_test = (df_test_intermed[feature_cols])

y_test_pred = logreg_1.predict(X_test)

cm_2 = confusion_matrix(y_test, y_test_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_2,
                              display_labels=logreg.classes_)
disp.plot()
plt.show()