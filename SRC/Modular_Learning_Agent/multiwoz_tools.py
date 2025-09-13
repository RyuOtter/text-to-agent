# Disclaimer: This file consists entirely of code from the AutoTOD codebase
###########################################################################

import random
import re
import sqlite3
from collections import OrderedDict
from functools import partial
from langchain_community.utilities import SQLDatabase
from sqlalchemy import Column, Integer, String, create_engine, or_
from sqlalchemy.orm import declarative_base, sessionmaker
from .multiwoz_data_utils import DB_PATH as MULTIWOZ_DB_PATH, BOOK_DB_PATH as MULTIWOZ_BOOK_DB_PATH, TableItem, clean_time, clean_name

# Create database query functions for a specific domain with SQL
def prepare_query_db_functions(domain, db_path=MULTIWOZ_DB_PATH):

    # Execute SQL query
    def query_db(sql, table=None, db_path=MULTIWOZ_DB_PATH):

        if table and table not in sql:
            return  f'Please query the {table} table in the database.'

        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute(sql)
        except Exception as e:
            return str(e)
        records = cursor.fetchall()

        if len(records) == 0:
            return 'No results found.'
        
        max_items = 5
        max_chars = 500

        result = []
        n_chars = 0
        line = '| ' + ' | '.join(desc[0] for desc in cursor.description) + ' |'
        n_chars += len(line) + 1
        result.append(line)
        line = '| ' + ' | '.join(['---'] * len(cursor.description)) + ' |'
        n_chars += len(line) + 1
        result.append(line)
        for i, record in enumerate(records, start=1):
            line = '| ' + ' | '.join(str(v) for v in record) + ' |'
            n_chars += len(line) + 1
            if n_chars <= max_chars and i <= max_items:
                result.append(line)
            else:
                n_left = len(records) - i + 1
                result.append(f'\n{n_left} more records ...')
                break
        result = '\n'.join(result)
        return result


    # Get database table schema information for a domain
    def get_table_info(domain, db_path):
        db = SQLDatabase.from_uri(
            database_uri=f'sqlite:///{db_path}',
            include_tables=[domain],
            sample_rows_in_table_info=2,
        )
        table_info = db.get_table_info()
        return table_info
    
    # Create function to create SQL query tools
    def make_schema(domain, name, table_info):
        func_desc_temp = '''Use an SQL statement to query the {domain} table to get required information.

        Table Schema:
        {table_info}'''

        param_desc_temp = f'The SQL statement to query the {domain} table.'
        
        schema = {
            'name': name,
            'description': func_desc_temp.format(table_info=table_info, domain=domain),
            'parameters': {
                'type': 'object',
                'properties': {
                    'sql': {
                        'type': 'string',
                        'description': param_desc_temp.format(domain=domain),
                    }
                },
                'required': ['sql'],
            }
        }
        return schema

    assert domain in ['restaurant', 'hotel', 'attraction', 'train']
    name = f'query_{domain}s'
    function = partial(query_db, table=domain)
    table_info = get_table_info(domain, db_path)
    schema = make_schema(domain, name, table_info)

    return {'name': name, 'function': function, 'schema': schema}



# Create booking function for restaurant, hotel, train or taxi
def prepare_book_functions(domain):
    if domain == 'restaurant':

        # Book restaurant
        def book_restaurant(name, people, day, time):
            info = {'name': name, 'people': str(people), 'day': day, 'time': time}
            flag, msg = make_booking_db('restaurant', info)
            return msg

        name = 'book_restaurant'
        schema = {
            'name': name,
            'description': 'Book a restaurant with certain requirements.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'name': {
                        'type': 'string',
                        'description': 'the name of the restaurant to book',
                    },
                    'people': {
                        'type': 'integer',
                        'description': 'the number of people',
                    },
                    'day': {
                        'type': 'string',
                        "enum": ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
                        'description': 'the day when the people go to the restaurant',
                    },
                    'time': {
                        'type': 'string',
                        'description': 'the time of the reservation',
                    },
                },
                'required': ['name', 'people', 'day', 'time'],
            }
        }
        return {'name': name, 'function': book_restaurant, 'schema': schema}

    elif domain == 'hotel':

        # Book hotel
        def book_hotel(name, people, day, stay):
            info = {'name': name, 'people': str(people), 'day': day, 'stay': str(stay)}
            flag, msg = make_booking_db('hotel', info)
            return msg

        name = 'book_hotel'
        schema = {
            'name': name,
            'description': 'Book a hotel with certain requirements.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'name': {
                        'type': 'string',
                        'description': 'the name of the hotel to book',
                    },
                    'people': {
                        'type': 'integer',
                        'description': 'the number of people',
                    },
                    'day': {
                        'type': 'string',
                        "enum": ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
                        'description': 'the day when the reservation starts',
                    },
                    'stay': {
                        'type': 'integer',
                        'description': 'the number of days of the reservation',
                    },
                },
                'required': ['name', 'people', 'day', 'stay'],
            }
        }
        return {'name': name, 'function': book_hotel, 'schema': schema}

    elif domain == 'train':

        # Buy train tickets
        def buy_train_tickets(train_id, tickets):
            info = {'train id': train_id, 'tickets': str(tickets)}
            flag, msg = make_booking_db('train', info)
            return msg

        name = 'buy_train_tickets'
        schema = {
            'name': name,
            'description': 'Buy train tickets.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'train_id': {
                        'type': 'string',
                        'description': 'the unique id of the train',
                    },
                    'tickets': {
                        'type': 'integer',
                        'description': 'the number of tickets to buy',
                    },
                },
                'required': ['train_id', 'tickets'],
            }
        }
        return {'name': name, 'function': buy_train_tickets, 'schema': schema}

    elif domain == 'taxi':

        # Book taxi
        def book_taxi(departure, destination, leave_time=None, arrive_time=None):
            info = {'departure': departure, 'destination': destination}
            if leave_time:
                info['leave time'] = leave_time
            if arrive_time:
                info['arrive time'] = arrive_time
            flag, msg = make_booking_taxi(info)
            return msg

        name = 'book_taxi'
        schema = {
            'name': name,
            'description': 'Book a taxi with certain requirements.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'departure': {
                        'type': 'string',
                        'description': 'the departure of the taxi',
                    },
                    'destination': {
                        'type': 'string',
                        'description': 'the destination of the taxi',
                    },
                    'leave_time': {
                        'type': 'string',
                        'description': 'the leave time of the taxi',
                    },
                    'arrive_time': {
                        'type': 'string',
                        'description': 'the arrive time of the taxi',
                    },
                },
                'required': ['departure', 'destination'],
            }
        }
        return {'name': name, 'function': book_taxi, 'schema': schema}
    
    else:
        raise ValueError("Wrong domain")


# Base class for booking records
class BookRecord(TableItem):

    # Check if booking record satisfies given constraints
    def satisfying(self, constraint):
        cons = {}
        for slot, value in constraint.items():
            if 'invalid' in slot:
                continue
            cons[slot] = value.lower()

        for slot, cons_value in cons.items():
            db_value = getattr(self, slot, None)
            if db_value != cons_value:
                return False
        else:
            return True
    

Base = declarative_base()


# Model for restaurant booking records
class RestaurantBook(Base, BookRecord):
    __tablename__ = 'restaurant_book'

    id = Column(Integer, primary_key=True)
    refer_number = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=False)
    people = Column(String, nullable=False)
    day = Column(String, nullable=False)
    time = Column(String, nullable=False)


# Model for hotel booking record
class HotelBook(Base, BookRecord):
    __tablename__ = 'hotel_book'

    id = Column(Integer, primary_key=True)
    refer_number = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=False)
    people = Column(String, nullable=False)
    day = Column(String, nullable=False)
    stay = Column(String, nullable=False)


# Model for train booking record
class TrainBook(Base, BookRecord):
    __tablename__ = 'train_book'

    id = Column(Integer, primary_key=True)
    refer_number = Column(String, nullable=False, unique=True)
    trainID = Column(String, nullable=False)
    tickets = Column(String, nullable=False)


# Create reference number for bookings
def generate_reference_num():
    return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for i in range(8))


DOMAIN_BOOK_CLASS_MAP = {
    'restaurant': RestaurantBook,
    'hotel': HotelBook,
    'train': TrainBook,
}


# Check if a value exists in a database column
def check_db_exist(table, column, value):
    conn = sqlite3.connect(MULTIWOZ_DB_PATH)
    sql = f'SELECT {column} FROM {table} WHERE {column} = "{value}"'
    result = conn.execute(sql)
    if result.fetchone():
        return True
    else:
        return False


# Query booking record by reference number
def query_booking_by_refer_num(domain, refer_number, book_db_path=MULTIWOZ_BOOK_DB_PATH):
    assert domain in DOMAIN_BOOK_CLASS_MAP

    engine = create_engine(f'sqlite:///{book_db_path}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    items = session.query(DOMAIN_BOOK_CLASS_MAP[domain])
    items = items.filter_by(refer_number=refer_number)
    item = items.first()
    return item


# Create booking in database with validation and return reference number
def make_booking_db(domain, info, book_db_path=MULTIWOZ_BOOK_DB_PATH):
    assert domain in DOMAIN_BOOK_CLASS_MAP

    info = {k: v.lower() for k, v in info.items()}

    DOMAIN_BOOK_SLOT_DESC = {
        'restaurant': {
            'name': 'the restaurant name',
            'people': 'the number of people',
            'day': 'the booking day',
            'time': 'the booking time',
        },
        'hotel': {
            'name': 'the hotel name',
            'people': 'the number of people',
            'day': 'the booking day',
            'stay': 'the days to stay',
        },
        'train': {
            'train id': 'the trian id',
            'tickets': 'the number of tickects',
        },
    }
    book_slot_desc = DOMAIN_BOOK_SLOT_DESC[domain]
    missing_slots = [slot for slot in book_slot_desc if slot not in info]
    if missing_slots != []:
        slots_str = ', '.join(book_slot_desc[s] for s in missing_slots)
        return False, f'Booking failed. Please provide {slots_str} for reservation.'
    
    if domain == 'restaurant':
        if info['name'] == '[restaurant name]':
            return False, f'Booking failed. Please provide the restaurant name to book.'
        missing_slots = [s for s in ['people', 'day', 'time'] if info[s] == '[value]']
        if len(missing_slots) > 0:
            missing_slots = ', '.join(missing_slots)
            return False, f'Booking failed. Please provide the values for {missing_slots}.'
        if not info['people'].isdigit() or int(info['people']) <= 0:
            return False, f'Booking failed. The value of people should be a positive integer.'
        DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        if info['day'] not in DAYS:
            return False, f'Booking failed. The value of day should be a day in a week.'
        info['time'] = clean_time(info['time'])
        if not re.fullmatch(r'\d\d:\d\d', info['time']):
            return False, f'Booking failed. please provide a valid time, like "08:30".'

    elif domain == 'hotel':
        if info['name'] == '[hotel name]':
            return False, f'Booking failed. Please provide the hotel name to book.'
        missing_slots = [s for s in ['people', 'day', 'stay'] if info[s] == '[value]']
        if len(missing_slots) > 0:
            missing_slots = ', '.join(missing_slots)
            return False, f'Booking failed. Please provide the values for {missing_slots}.'
        if not info['people'].isdigit() or int(info['people']) <= 0:
            return False, f'Booking failed. The value of people should be a positive integer.'
        DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        if info['day'] not in DAYS:
            return False, f'Booking failed. The value of day should be a day in a week.'
        if not info['stay'].isdigit() or int(info['people']) <= 0:
            return False, f'Booking failed. The value of stay should be a positive integer.'

    elif domain == 'train':
        if info['train id'] == '[train id]':
            return False, f'Booking failed. Please provide the train id to book.'
        if info['tickets'] == '[value]':
            return False, f'Booking failed. Please the number of tickets to book.'
        if not info['tickets'].isdigit() or int(info['tickets']) <= 0:
            return False, f'Booking failed. The value of tickets should be a positive integer.'
    else:
        raise ValueError("Wrong domain")

    if domain == 'restaurant':
        if not check_db_exist('restaurant', 'name', info['name']):
            return False, f'Booking failed. "{info["name"]}" is not found in the restaurant database. Please provide a valid restaurant name.'
    elif domain == 'hotel':
        if not check_db_exist('hotel', 'name', info['name']):
            return False, f'Booking failed. "{info["name"]}" is not found in the hotel database. Please provide a valid hotel name.'
    elif domain == 'train':
        if not check_db_exist('train', 'trainID', info['train id']):
            return False, f'Booking failed. "{info["train id"]}" is not found in the train databse. Please provide a valid train id.'
    else:
        raise ValueError("Wrong domain")

    engine = create_engine(f'sqlite:///{book_db_path}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    refer_number = generate_reference_num()
    if domain == 'train':
        info['trainID'] = info.pop('train id')
    book = DOMAIN_BOOK_CLASS_MAP[domain](refer_number=refer_number, **info)

    session.add(book)
    session.commit()

    return True, f'Booking succeed. The reference number is {refer_number}.'



# Generate taxi details for booking confirmation
def pick_taxi():
    taxi_colors = ["black","white","red","yellow","blue",'grey']
    taxi_types = ["toyota","skoda","bmw",'honda','ford','audi','lexus','volvo','volkswagen','tesla']

    color = random.choice(taxi_colors)
    brand = random.choice(taxi_types)
    phone = ''.join(random.choice('0123456789') for i in range(10))

    return color, brand, phone


# Create taxi booking
def make_booking_taxi(info):
    place_slots = {'departure': None, 'destination': None}
    missing_slots = [slot for slot in place_slots if info.get(slot) in [None, '[value]']]
    if missing_slots != []:
        slots_str = ' and '.join(missing_slots)
        return False, f'Booking failed. The {slots_str} is missing.'

    invalid_slots = []
    for slot in place_slots:
        place = info[slot]
        for domain in ['restaurant', 'hotel', 'attraction']:
            if venue := query_venue_by_name_or_address(domain, place):
                place_slots[slot] = venue
                break
        if not place_slots[slot]:
            invalid_slots.append(slot)
    if invalid_slots != []:
        slots_str = ' and '.join(invalid_slots)
        return False, f'Booking failed. Please provide valid place for the {slots_str}.'
    
    if place_slots['departure'].name == place_slots['destination'].name:
        return False, f'Booking failed. The departure and destination can not be the same place.'

    info2 = {}
    for k, v in info.items():
        if k == 'leave time':
            k = 'leave'
        elif k == 'arrive time':
            k = 'arrive'
        info2[k] = v
    info = info2
    time_slots = ['leave', 'arrive']

    present_slots = [slot for slot in time_slots if info.get(slot) not in [None, '[value]']]
    if len(present_slots) == 0:
        return False, f'Booking failed. The leave time or arrive time is missing.'
    
    invalid_slots = [slot for slot in present_slots if not re.fullmatch(r'\d\d:\d\d', clean_time(info[slot]))]
    if invalid_slots != []:
        slots_str = ' and '.join(invalid_slots)
        return False, f'Booking failed. Please provide valid time format for the {slots_str}, like "07:30".'

    color, brand, phone = pick_taxi()
    return True, f'Booking succeed. There is a {color} {brand} taxi. Contact number is {phone}.'


# Initialize base class for venue records
Base = declarative_base()

# Base class for venue records
class Venue(TableItem):

    # Check if venue satisfies requests
    def satisfying(self, constraint):
        cons = {}
        for slot, value in constraint.items():
            if value in ['dontcare', '', 'none', 'not mentioned']:
                continue
            if slot in ['postcode', 'phone']:
                value = ''.join(x for x in value if x != ' ')
                cons[slot] = value.lower()
            if slot == 'entrance fee':
                cons['entrance_fee'] = value
            else:
                cons[slot] = value.lower()

        for slot, cons_value in cons.items():
            db_value = getattr(self, slot, None)
            if db_value is None:
                return False
            db_value = db_value.lower()
            if slot == 'address':
                if not (db_value in cons_value):
                    return False
            else: 
                if db_value != cons_value:
                    return False
        else:
            return True

# Model for restaurant venues
class Restaurant(Base, Venue):
    __tablename__ = 'restaurant'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    area = Column(String)
    pricerange = Column(String)
    food = Column(String)
    phone = Column(String)
    postcode = Column(String)
    address = Column(String)

    # Return restaurant attributes
    def items(self):
        return OrderedDict(
            name=self.name,
            area=self.area,
            pricerange=self.pricerange,
            food=self.food,
            phone=self.phone,
            postcode=self.postcode,
            address=self.address,
        ).items()

# Model for hotel venues
class Hotel(Base, Venue):
    __tablename__ = 'hotel'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)
    area = Column(String)
    internet = Column(String)
    parking = Column(String)
    pricerange = Column(String)
    stars = Column(String)
    phone = Column(String)
    address = Column(String)
    postcode = Column(String)

    # Return hotel attributes
    def items(self):
        return OrderedDict(
            name=self.name,
            type=self.type,
            area=self.area,
            internet=self.internet,
            parking=self.parking,
            pricerange=self.pricerange,
            stars=self.stars,
            phone=self.phone,
            address=self.address,
            postcode=self.postcode,
        ).items()

# Model for attraction venues
class Attraction(Base, Venue):
    __tablename__ = 'attraction'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)
    area = Column(String)
    phone = Column(String)
    address = Column(String)
    postcode = Column(String)
    entrance_fee = Column(String)

    # Return attraction attributes
    def items(self):
        return OrderedDict(
            name=self.name,
            type=self.type,
            area=self.area,
            phone=self.phone,
            address=self.address,
            postcode=self.postcode,
            entrance_fee=self.entrance_fee,
        ).items()

# Model for train schedules
class Train(Base, Venue):
    __tablename__ = 'train'

    id = Column(Integer, primary_key=True)
    arriveBy = Column(String)
    day = Column(String)
    departure = Column(String)
    destination = Column(String)
    leaveAt = Column(String)
    price = Column(String)
    trainID = Column(String)
    duration = Column(String)

    # Return train schedule attributes
    def items(self):
        return OrderedDict(
            trainID=self.trainID,
            departure=self.departure,
            destination=self.destination,
            day=self.day,
            leaveAt=self.leaveAt,
            arriveBy=self.arriveBy,
            price=self.price,
            duration=self.duration,
        ).items()

    # Check if train satisfies time and route
    def satisfying(self, constraint: dict):
        for slot, cons_value in constraint.items():
            train_value = getattr(self, slot, None)
            if train_value is None:
                return False
            elif slot == 'leaveAt':
                if train_value < cons_value:
                    return False
            elif slot == 'arriveBy':
                if train_value > cons_value:
                    return False
            else:
                if train_value != cons_value:
                    return False
        return True
    

DOMAIN_CLASS_MAP = {
    'restaurant': Restaurant,
    'hotel': Hotel,
    'attraction': Attraction,
    'train': Train,
}

# Query venue by name or address
def query_venue_by_name_or_address(domain, place, db_path=MULTIWOZ_DB_PATH):
    assert domain in DOMAIN_CLASS_MAP
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    Venue = DOMAIN_CLASS_MAP[domain]
    items = session.query(Venue)
    items = items.filter(or_(Venue.name == clean_name(place), Venue.address == place))
    item = items.first()
    return item


# Query train schedule
def query_train_by_id(id, db_path=MULTIWOZ_DB_PATH):
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    items = session.query(Train)
    items = items.filter_by(trainID=id)
    item = items.first()
    return item


# Query trains with constraints
def query_trains(info, db_path=MULTIWOZ_DB_PATH):
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    items = session.query(Train)

    sub_info = {s: info[s] for s in ['day', 'departure', 'destination', 'trainID'] if s in info}
    items = items.filter_by(**sub_info)
    if time := info.get('leaveAt'):
        items = items.filter(Train.leaveAt >= time)
    if time := info.get('arriveBy'):
        items = items.filter(Train.arriveBy <= time)

    return items.all()


# Query venue by name
def query_venue_by_name(domain, name, db_path=MULTIWOZ_DB_PATH):
    assert domain in DOMAIN_CLASS_MAP

    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    items = session.query(DOMAIN_CLASS_MAP[domain])
    name = clean_name(name)
    items = items.filter_by(name=name)
    item = items.first()
    return item

# Execute SQL query
def query_by_sql(sql, db_path=MULTIWOZ_DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(sql)
    records = cursor.fetchall()
    return records