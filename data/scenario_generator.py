import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

class CrisisDataSimulator:
    def __init__(self, n_points=500, days=30):
        """Initialize the simulator with time parameters"""
        self.n_points = n_points
        self.days = days
        self.time = np.linspace(0, days, n_points)
        # Create actual dates for x-axis
        self.dates = [datetime.now() + timedelta(days=x*(days/n_points)) for x in range(n_points)]
        
        # Define crisis event time (when the major event happens)
        self.event_day = days * 0.6  # Event at 60% of the time series
    
    def add_noise(self, data, scale=0.05):
        """Add proportional noise to the data"""
        noise_scale = scale * (np.max(data) - np.min(data))
        return data + np.random.normal(0, noise_scale, len(data))
    
    def sigmoid_transition(self, start_val, end_val, transition_center, steepness=5):
        """Create a sigmoid transition between values"""
        factor = 1 / (1 + np.exp(-steepness * (self.time - transition_center)))
        return start_val + (end_val - start_val) * factor

    def simulate_financial_crisis(self):
        """Simulate data for financial market crisis"""
        # Stock prices: gradual decline followed by sharp drop
        stock_prices = 100 - 0.5 * self.time
        # Add crash after event
        crash_factor = np.ones_like(self.time)
        crash_factor[self.time > self.event_day] = 1 - 0.2 * (self.time[self.time > self.event_day] - self.event_day)
        stock_prices = stock_prices * crash_factor
        stock_prices = self.add_noise(stock_prices)
        
        # Trading volume: spike dramatically during crash
        trading_volume = 50 + 5 * np.sin(self.time / 3)  # Normal fluctuations
        # Add volume spike
        volume_spike = np.zeros_like(self.time)
        mask = (self.time > self.event_day) & (self.time < self.event_day + 5)
        spike_time = self.time[mask] - self.event_day
        volume_spike[mask] = 200 * np.exp(-((spike_time - 1) ** 2) / 2)
        trading_volume = trading_volume + volume_spike
        trading_volume = self.add_noise(trading_volume, 0.1)
        
        # Interest rates: sharp increase as central banks respond
        interest_rates = 2.5 * np.ones_like(self.time)
        interest_rates[self.time > self.event_day] = 2.5 + 2 * (1 - np.exp(-(self.time[self.time > self.event_day] - self.event_day)))
        interest_rates = self.add_noise(interest_rates, 0.03)
        
        # Consumer confidence: plummeting
        consumer_confidence = 75 - 0.3 * self.time
        confidence_drop = np.zeros_like(self.time)
        confidence_drop[self.time > self.event_day] = 30 * (1 - np.exp(-(self.time[self.time > self.event_day] - self.event_day) / 2))
        consumer_confidence = consumer_confidence - confidence_drop
        consumer_confidence = self.add_noise(consumer_confidence)
        
        # Manufacturing output: declining
        manufacturing = 100 - 0.2 * self.time
        manufacturing[self.time > self.event_day] = manufacturing[self.time > self.event_day] - 15 * (1 - np.exp(-(self.time[self.time > self.event_day] - self.event_day) / 3))
        manufacturing = self.add_noise(manufacturing)
        
        # Credit spreads: widening rapidly during crisis
        credit_spreads = 2 + 0.05 * self.time
        credit_spreads[self.time > self.event_day] = credit_spreads[self.time > self.event_day] + 5 * (1 - np.exp(-(self.time[self.time > self.event_day] - self.event_day) / 1.5))
        credit_spreads = self.add_noise(credit_spreads, 0.1)
        
        data = {
            'Stock Prices': stock_prices,
            'Trading Volume': trading_volume,
            'Interest Rates': interest_rates,
            'Consumer Confidence': consumer_confidence,
            'Manufacturing Output': manufacturing,
            'Credit Spreads': credit_spreads
        }
        
        title = "Financial Market Crisis Indicators"
        explanation = """
        This simulation shows a financial market crash occurring around day {:.0f}.
        The pattern indicates a severe financial Crisis with panic selling (volume spike),
        emergency rate hikes, collapsing confidence, and production slowdown.
        Such patterns were observed during the 2008 financial crisis and other market crashes.
        """
        
        return data, title, explanation.format(self.event_day)

    def simulate_sepsis(self):
        """Simulate patient data showing development of sepsis"""
        # Heart rate: gradually increasing, then spiking
        normal_hr = 70 + 5 * np.sin(self.time / 2)  # Natural oscillation
        heart_rate = normal_hr.copy()
        # Add sepsis progression
        heart_rate[self.time > self.event_day-5] += 0.8 * (self.time[self.time > self.event_day-5] - (self.event_day-5)) ** 2
        heart_rate = self.add_noise(heart_rate, 0.03)
        
        # Blood pressure: initially stable, then dropping
        blood_pressure = 120 + 5 * np.sin(self.time / 1.5)  # Normal fluctuations
        # Add septic shock drop
        bp_drop = np.zeros_like(self.time)
        bp_drop[self.time > self.event_day] = 40 * (1 - np.exp(-(self.time[self.time > self.event_day] - self.event_day) / 3))
        blood_pressure = blood_pressure - bp_drop
        blood_pressure = self.add_noise(blood_pressure, 0.04)
        
        # Body temperature: rising with infection, then potentially dropping in severe sepsis
        temperature = 37 + 0.2 * np.sin(self.time / 3)  # Normal daily fluctuations
        # Add fever spike and then drop
        fever_pattern = np.zeros_like(self.time)
        fever_pattern[self.time > self.event_day-7] = 2 * np.exp(-((self.time[self.time > self.event_day-7] - (self.event_day+1)) ** 2) / 20)
        temperature = temperature + fever_pattern
        temperature = self.add_noise(temperature, 0.02)
        
        # Respiratory rate: increasing with sepsis
        resp_rate = 14 + np.sin(self.time / 2)  # Normal breathing
        resp_increase = np.zeros_like(self.time)
        resp_increase[self.time > self.event_day-6] = 8 * (1 - np.exp(-(self.time[self.time > self.event_day-6] - (self.event_day-6)) / 5))
        resp_rate = resp_rate + resp_increase
        resp_rate = self.add_noise(resp_rate, 0.05)
        
        # Oxygen saturation: gradually decreasing
        oxygen = 98 - 0.1 * self.time  # Slight baseline decrease
        # Add sepsis-related drop
        oxygen_drop = np.zeros_like(self.time)
        oxygen_drop[self.time > self.event_day-3] = 15 * (1 - np.exp(-(self.time[self.time > self.event_day-3] - (self.event_day-3)) / 6))
        oxygen = oxygen - oxygen_drop
        oxygen = np.clip(oxygen, 70, 100)  # Clip to realistic range
        oxygen = self.add_noise(oxygen, 0.01)
        
        # White blood cell count: abnormal pattern in sepsis
        wbc = 7 + np.sin(self.time / 4)  # Normal fluctuations
        # Abnormal response - initially very high, potentially dropping in severe sepsis
        wbc_response = np.zeros_like(self.time)
        wbc_response[self.time > self.event_day-8] = 10 * np.exp(-((self.time[self.time > self.event_day-8] - (self.event_day-2)) ** 2) / 40)
        wbc = wbc + wbc_response
        wbc = self.add_noise(wbc, 0.07)
        
        data = {
            'Heart Rate': heart_rate,
            'Blood Pressure': blood_pressure,
            'Body Temperature': temperature,
            'Respiratory Rate': resp_rate,
            'Oxygen Saturation': oxygen,
            'WBC Count': wbc
        }
        
        title = "Patient Sepsis Development Indicators"
        explanation = """
        This simulation shows the development of sepsis in a patient, with critical deterioration around day {:.0f}.
        The pattern shows classic sepsis progression: initial infection with fever and elevated heart/respiratory rates,
        followed by shock with dropping blood pressure and potential organ failure.
        Early detection of these combined patterns can be life-saving.
        """
        
        return data, title, explanation.format(self.event_day)

    def simulate_drought(self):
        """Simulate environmental data showing severe drought impact"""
        # Precipitation: consistently below normal with occasional minimal rainfall
        precip = 50 - 0.8 * self.time  # Declining trend
        # Add occasional small rain events
        for i in range(5):
            center = np.random.uniform(0, self.days)
            height = np.random.uniform(5, 15)
            width = np.random.uniform(0.2, 0.5)
            precip += height * np.exp(-((self.time - center) / width) ** 2) 
        precip = np.clip(precip, 0, 100)  # Can't have negative precipitation
        precip = self.add_noise(precip, 0.1)
        
        # Soil moisture: steadily decreasing
        soil_moisture = 80 - 1.5 * self.time
        soil_moisture = np.clip(soil_moisture, 10, 100)
        soil_moisture = self.add_noise(soil_moisture, 0.05)
        
        # River flow rates: approaching record lows
        river_flow = 70 - 1.2 * self.time
        # Add normal seasonal pattern
        river_flow += 10 * np.sin(2 * np.pi * self.time / 30)
        river_flow = np.clip(river_flow, 5, 100)
        river_flow = self.add_noise(river_flow, 0.07)
        
        # Vegetation health: declining
        veg_health = 90 - 0.7 * self.time
        # Accelerated decline after prolonged drought
        veg_health[self.time > self.event_day] -= 1.5 * (self.time[self.time > self.event_day] - self.event_day)
        veg_health = np.clip(veg_health, 0, 100)
        veg_health = self.add_noise(veg_health, 0.08)
        
        # Wildlife movement: changing patterns (increasing distance from normal ranges)
        wildlife_movement = 10 + 0.8 * self.time
        # Sharp increase as animals search for water
        wildlife_movement[self.time > self.event_day-5] += 2 * (self.time[self.time > self.event_day-5] - (self.event_day-5))
        wildlife_movement = self.add_noise(wildlife_movement, 0.15)
        
        # Fire risk: rising sharply
        fire_risk = 30 + 1.2 * self.time
        # Accelerated increase during peak drought
        fire_risk[self.time > self.event_day-3] += 3 * (self.time[self.time > self.event_day-3] - (self.event_day-3))
        fire_risk = np.clip(fire_risk, 0, 100)
        fire_risk = self.add_noise(fire_risk, 0.06)
        
        data = {
            'Precipitation': precip,
            'Soil Moisture': soil_moisture,
            'River Flow': river_flow,
            'Vegetation Health': veg_health,
            'Wildlife Movement': wildlife_movement,
            'Fire Risk': fire_risk
        }
        
        title = "Severe Drought Impact Indicators"
        explanation = """
        This simulation shows a worsening drought with critical ecological impacts around day {:.0f}.
        The combined patterns indicate a severe drought with cascading effects: depleted water resources,
        stressed vegetation, wildlife behavioral changes, and extreme fire danger.
        These patterns mirror real drought emergencies like those in California (2012-2016) and Australia (2017-2019).
        """
        
        return data, title, explanation.format(self.event_day)

    def simulate_crop_disease(self):
        """Simulate agricultural data showing crop disease outbreak"""
        # Crop health: declining rapidly after infection
        crop_health = 95 * np.ones_like(self.time)
        # Disease progression
        disease_impact = np.zeros_like(self.time)
        disease_impact[self.time > self.event_day-7] = 80 * (1 - np.exp(-(self.time[self.time > self.event_day-7] - (self.event_day-7)) / 5))
        crop_health = crop_health - disease_impact
        crop_health = np.clip(crop_health, 0, 100)
        crop_health = self.add_noise(crop_health, 0.05)
        
        # Moisture levels: normal or high (conducive to fungal growth)
        moisture = 70 + 10 * np.sin(self.time / 5)  # Normal oscillation with rain events
        moisture = self.add_noise(moisture, 0.08)
        
        # Temperature: in optimal range for fungal growth
        temperature = 25 + 3 * np.sin(2 * np.pi * self.time / self.days)  # Daily cycle
        temperature = self.add_noise(temperature, 0.04)
        
        # Pest detection: spikes in specific areas
        pest_levels = 5 + 2 * np.sin(self.time / 4)  # Normal fluctuations
        # Disease vectors increasing
        pest_increase = np.zeros_like(self.time)
        pest_increase[self.time > self.event_day-10] = 20 * (1 - np.exp(-(self.time[self.time > self.event_day-10] - (self.event_day-10)) / 3))
        pest_levels = pest_levels + pest_increase
        pest_levels = self.add_noise(pest_levels, 0.2)
        
        # Soil nutrient levels: normal (ruling out deficiency)
        nutrients = 85 + 5 * np.sin(self.time / 10)
        nutrients = self.add_noise(nutrients, 0.03)
        
        # UV radiation: below normal (less sunlight, more humidity)
        uv_levels = 60 - 0.5 * self.time
        uv_levels += 15 * np.sin(2 * np.pi * self.time / self.days)  # Daily cycle
        uv_levels = np.clip(uv_levels, 20, 100)
        uv_levels = self.add_noise(uv_levels, 0.06)
        
        data = {
            'Crop Health': crop_health,
            'Moisture Levels': moisture,
            'Temperature': temperature,
            'Pest Detection': pest_levels,
            'Soil Nutrients': nutrients,
            'UV Radiation': uv_levels
        }
        
        title = "Crop Disease Outbreak Indicators"
        explanation = """
        This simulation shows a fungal disease outbreak affecting crops, becoming critical around day {:.0f}.
        The combined data indicates ideal conditions for fungal pathogens: adequate moisture, 
        optimal temperatures, reduced sunlight, and increasing pest vectors, while normal nutrient levels
        rule out deficiency as a cause. Similar patterns occur in potato blight or wheat rust outbreaks.
        """
        
        return data, title, explanation.format(self.event_day)

    def simulate_infrastructure_failure(self):
        """Simulate transportation data showing major infrastructure failure"""
        # Traffic flow: drops to zero in affected sector
        traffic_flow = 80 + 15 * np.sin(2 * np.pi * self.time / 1)  # Daily cycle
        # Infrastructure failure
        traffic_flow[self.time > self.event_day] = 5 + 5 * np.random.random(sum(self.time > self.event_day))
        traffic_flow = self.add_noise(traffic_flow, 0.07)
        
        # Rerouted traffic: spikes in adjacent sectors
        rerouted = 30 + 10 * np.sin(2 * np.pi * self.time / 1)  # Normal daily pattern
        # Traffic rerouting after failure
        reroute_spike = np.zeros_like(self.time)
        reroute_spike[self.time > self.event_day] = 150 * np.exp(-(self.time[self.time > self.event_day] - self.event_day) / 3)
        rerouted = rerouted + reroute_spike
        rerouted = self.add_noise(rerouted, 0.1)
        
        # Emergency vehicle activity: spike after incident
        emergency = 10 + 5 * np.random.random(self.n_points)  # Random normal activity
        # Emergency response
        emergency_spike = np.zeros_like(self.time)
        emergency_spike[self.time > self.event_day] = 80 * np.exp(-(self.time[self.time > self.event_day] - self.event_day) / 2)
        emergency = emergency + emergency_spike
        emergency = self.add_noise(emergency, 0.15)
        
        # Public transit rerouting: active after failure
        transit_changes = np.zeros_like(self.time)
        transit_changes[self.time > self.event_day] = 100 * (1 - np.exp(-(self.time[self.time > self.event_day] - self.event_day) / 10))
        transit_changes = self.add_noise(transit_changes, 0.05)
        
        # Social media activity about transportation: spikes dramatically
        social_media = 20 + 10 * np.random.random(self.n_points)  # Random baseline chatter
        # Viral spread of incident news
        sm_spike = np.zeros_like(self.time)
        sm_spike[self.time > self.event_day] = 200 * np.exp(-(self.time[self.time > self.event_day] - self.event_day) / 1.5)
        social_media = social_media + sm_spike
        social_media = self.add_noise(social_media, 0.2)
        
        # Weather conditions: normal (ruling out weather causes)
        weather = 70 + 10 * np.sin(2 * np.pi * self.time / 1)  # Normal daily pattern
        weather = self.add_noise(weather, 0.03)
        
        data = {
            'Traffic Flow': traffic_flow,
            'Rerouted Traffic': rerouted,
            'Emergency Vehicles': emergency,
            'Transit Rerouting': transit_changes,
            'Social Media Activity': social_media,
            'Weather Conditions': weather
        }
        
        title = "Major Infrastructure Failure Indicators"
        explanation = """
        This simulation shows a major infrastructure failure occurring on day {:.0f}.
        The pattern indicates a bridge collapse or similar critical failure: traffic flow
        suddenly stops, emergency vehicles converge, adjacent routes become congested,
        transit systems implement emergency protocols, and social media activity explodes.
        Weather remains normal, ruling out environmental causes.
        """
        
        return data, title, explanation.format(self.event_day)

    def simulate_machine_failure(self):
        """Simulate manufacturing data showing production line critical failure"""
        # Machine vibration: increasing, then spike, then flatline
        vibration = 20 + 0.7 * self.time
        # Add pre-failure increase and post-failure flatline
        vibration[self.time > self.event_day-5] += 5 * (self.time[self.time > self.event_day-5] - (self.event_day-5))
        # Massive spike at failure
        failure_spike = np.zeros_like(self.time)
        failure_mask = (self.time > self.event_day) & (self.time < self.event_day + 0.5)
        failure_spike[failure_mask] = 200
        vibration = vibration + failure_spike
        # Flatline after failure
        vibration[self.time > self.event_day + 0.5] = 0
        vibration = self.add_noise(vibration, 0.08)
        
        # Production output: dropping after failure
        production = 100 * np.ones_like(self.time)
        # Minor quality issues before failure
        production[self.time > self.event_day-7] -= 1 * (self.time[self.time > self.event_day-7] - (self.event_day-7))
        # Complete production stop after failure
        production[self.time > self.event_day] = 0
        production = self.add_noise(production, 0.05)
        
        # Energy consumption: abnormal pattern, spike, then drop
        energy = 50 + 10 * np.sin(2 * np.pi * self.time / 1)  # Normal cycle
        # Increasing energy use as machine struggles
        energy[self.time > self.event_day-6] += 3 * (self.time[self.time > self.event_day-6] - (self.event_day-6))
        # Spike during failure
        energy_spike = np.zeros_like(self.time)
        energy_spike_mask = (self.time > self.event_day) & (self.time < self.event_day + 0.3)
        energy_spike[energy_spike_mask] = 150
        energy = energy + energy_spike
        # Drop to minimal after failure
        energy[self.time > self.event_day + 0.3] = 10
        energy = self.add_noise(energy, 0.07)
        
        # Temperature: rising before failure, spike during failure
        temperature = 60 + 5 * np.sin(2 * np.pi * self.time / 1)  # Normal daily cycle
        # Rising as problems develop
        temperature[self.time > self.event_day-4] += 4 * (self.time[self.time > self.event_day-4] - (self.event_day-4))
        # Spike during failure
        temp_spike = np.zeros_like(self.time)
        temp_spike_mask = (self.time > self.event_day) & (self.time < self.event_day + 0.4)
        temp_spike[temp_spike_mask] = 100
        temperature = temperature + temp_spike
        # Cool down after shutdown
        temperature[self.time > self.event_day + 0.4] = 30 + 0.2 * (self.time[self.time > self.event_day + 0.4] - (self.event_day + 0.4))
        temperature = self.add_noise(temperature, 0.06)
        
        # Quality control: increasing failures before event
        quality_failures = 5 + 0.1 * self.time
        # Rising failures as machine deteriorates
        quality_failures[self.time > self.event_day-10] += 2 * (self.time[self.time > self.event_day-10] - (self.event_day-10))
        # No production, so no failures after breakdown
        quality_failures[self.time > self.event_day] = 0
        quality_failures = self.add_noise(quality_failures, 0.2)
        
        # Maintenance schedule: overdue status increasing
        maintenance_overdue = 10 + 2 * self.time
        maintenance_overdue = np.clip(maintenance_overdue, 0, 100)
        # Flatline after failure (machine offline for major repair)
        maintenance_overdue[self.time > self.event_day] = 0
        maintenance_overdue = self.add_noise(maintenance_overdue, 0.04)
        
        data = {
            'Machine Vibration': vibration,
            'Production Output': production,
            'Energy Consumption': energy,
            'Temperature': temperature,
            'Quality Failures': quality_failures,
            'Maintenance Overdue': maintenance_overdue
        }
        
        title = "Production Line Critical Failure Indicators"
        explanation = """
        This simulation shows a catastrophic machine failure occurring on day {:.0f}.
        The pattern reveals a preventable failure with multiple warning signs: increasing vibration,
        rising temperature, worsening quality issues, and overdue maintenance.
        The failure sequence includes energy spike, temperature surge, and complete production stoppage.
        Similar patterns occur before major industrial equipment failures across manufacturing sectors.
        """
        
        return data, title, explanation.format(self.event_day)
    
    def simulate_cloud_app_performance(self):
        """Simulate data for cloud application performance degradation"""
        # API response times: gradual increase with sharp spike
        api_response = 100 + 0.5 * self.time  # Gradual baseline increase
        # Add spike after network bottleneck appears
        response_spike = np.zeros_like(self.time)
        response_spike[self.time > self.event_day-2] = 150 * (1 - np.exp(-(self.time[self.time > self.event_day-2] - (self.event_day-2))))
        api_response = api_response + response_spike
        api_response = self.add_noise(api_response, 0.1)
        
        # Database latency: increasing under load
        db_latency = 50 + 0.3 * self.time  # Slight baseline increase
        # Add spike when connection pool is exhausted
        db_spike = np.zeros_like(self.time)
        db_spike[self.time > self.event_day] = 300 * (1 - np.exp(-(self.time[self.time > self.event_day] - self.event_day) / 0.5))
        db_latency = db_latency + db_spike
        db_latency = self.add_noise(db_latency, 0.15)
        
        # CPU utilization: high but not critical
        cpu_usage = 60 + 5 * np.sin(self.time / 2)  # Normal fluctuations
        # Add mild increase under load
        cpu_usage[self.time > self.event_day-3] += 10 * (1 - np.exp(-(self.time[self.time > self.event_day-3] - (self.event_day-3)) / 2))
        cpu_usage = np.clip(cpu_usage, 0, 100)  # Clip to realistic range
        cpu_usage = self.add_noise(cpu_usage, 0.05)
        
        # Network throughput: decreasing between services
        network_throughput = 90 - 0.2 * self.time  # Slight decrease over time
        # Add significant drop during bottleneck
        throughput_drop = np.zeros_like(self.time)
        throughput_drop[self.time > self.event_day-1] = 40 * (1 - np.exp(-(self.time[self.time > self.event_day-1] - (self.event_day-1)) / 0.5))
        network_throughput = network_throughput - throughput_drop
        network_throughput = np.clip(network_throughput, 0, 100)
        network_throughput = self.add_noise(network_throughput, 0.08)
        
        # Error rates: increasing for specific services
        error_rates = 2 + 0.05 * self.time  # Low baseline error rate
        # Add error rate spike during incident
        error_spike = np.zeros_like(self.time)
        error_spike[self.time > self.event_day] = 25 * (1 - np.exp(-(self.time[self.time > self.event_day] - self.event_day) / 0.8))
        error_rates = error_rates + error_spike
        error_rates = self.add_noise(error_rates, 0.2)
        
        # Connection pool usage: depleting rapidly
        conn_pool = 60 + 0.5 * self.time  # Gradual increase over time
        # Add spike as connections are exhausted
        conn_spike = np.zeros_like(self.time)
        conn_spike[self.time > self.event_day-1.5] = 35 * (1 - np.exp(-(self.time[self.time > self.event_day-1.5] - (self.event_day-1.5)) / 0.6))
        conn_pool = conn_pool + conn_spike
        conn_pool = np.clip(conn_pool, 0, 100)
        conn_pool = self.add_noise(conn_pool, 0.07)
        
        data = {
            'API Response Time (ms)': api_response,
            'Database Latency (ms)': db_latency,
            'CPU Utilization (%)': cpu_usage,
            'Network Throughput (%)': network_throughput,
            'Error Rate (%)': error_rates,
            'Connection Pool Usage (%)': conn_pool
        }
        
        title = "Cloud Application Performance Degradation Indicators"
        explanation = """
        This simulation shows a network bottleneck causing service degradation starting around day {:.0f}.
        The pattern indicates what appears as a database issue is actually a network problem between services.
        Note how CPU utilization remains reasonable while network throughput drops significantly.
        Connection pool exhaustion follows, causing database latency spikes and increasing error rates.
        This pattern is typical of microservice architecture issues where network partitioning occurs.
        """
        
        return data, title, explanation.format(self.event_day)

    def simulate_ecommerce_sales_event(self):
        """Simulate e-commerce platform metrics during a major sales event"""
        # Transaction volume: extreme spike
        transactions = 100 + 20 * np.sin(self.time / 3)  # Normal daily patterns
        # Add sales event spike
        trans_spike = np.zeros_like(self.time)
        trans_spike[self.time > self.event_day] = 400 * np.exp(-(self.time[self.time > self.event_day] - self.event_day) ** 2 / 15)
        transactions = transactions + trans_spike
        transactions = self.add_noise(transactions, 0.12)
        
        # Cache hit ratio: collapsing under load
        cache_hit = 85 - 0.2 * self.time  # Slight decrease over time
        # Add severe drop during event
        cache_drop = np.zeros_like(self.time)
        cache_drop[self.time > self.event_day] = 45 * (1 - np.exp(-(self.time[self.time > self.event_day] - self.event_day) / 0.5))
        cache_hit = cache_hit - cache_drop
        cache_hit = np.clip(cache_hit, 0, 100)
        cache_hit = self.add_noise(cache_hit, 0.08)
        
        # Payment gateway response: timeouts increasing
        payment_time = 200 + 20 * np.sin(self.time / 2)  # Normal variations
        # Add severe timeout increases
        payment_spike = np.zeros_like(self.time)
        payment_spike[self.time > self.event_day+0.5] = 800 * (1 - np.exp(-(self.time[self.time > self.event_day+0.5] - (self.event_day+0.5)) / 1))
        payment_time = payment_time + payment_spike
        payment_time = self.add_noise(payment_time, 0.15)
        
        # Customer session duration: shortening as users abandon
        session_duration = 600 - 3 * self.time  # Gradual decrease
        # Add sharp drop during failures
        session_drop = np.zeros_like(self.time)
        session_drop[self.time > self.event_day+1] = 350 * (1 - np.exp(-(self.time[self.time > self.event_day+1] - (self.event_day+1)) / 0.8))
        session_duration = session_duration - session_drop
        session_duration = np.clip(session_duration, 50, 700)
        session_duration = self.add_noise(session_duration, 0.07)
        
        # Cart abandonment: spiking as system struggles
        cart_abandon = 25 + 5 * np.sin(self.time / 4)  # Normal patterns
        # Add spike during failures
        abandon_spike = np.zeros_like(self.time)
        abandon_spike[self.time > self.event_day+1] = 45 * (1 - np.exp(-(self.time[self.time > self.event_day+1] - (self.event_day+1)) / 0.8))
        cart_abandon = cart_abandon + abandon_spike
        cart_abandon = np.clip(cart_abandon, 0, 100)
        cart_abandon = self.add_noise(cart_abandon, 0.06)
        
        # Infrastructure scaling: hitting limits
        infra_scaling = 40 + 0.8 * self.time  # Gradual increase over time
        # Add max-out during event
        scaling_spike = np.zeros_like(self.time)
        scaling_spike[self.time > self.event_day] = 50 * (1 - np.exp(-(self.time[self.time > self.event_day] - self.event_day) / 0.3))
        infra_scaling = infra_scaling + scaling_spike
        infra_scaling = np.clip(infra_scaling, 0, 100)
        infra_scaling = self.add_noise(infra_scaling, 0.05)
        
        data = {
            'Transaction Volume': transactions,
            'Cache Hit Ratio (%)': cache_hit,
            'Payment Gateway Response (ms)': payment_time,
            'Session Duration (sec)': session_duration,
            'Cart Abandonment Rate (%)': cart_abandon,
            'Infrastructure Scaling (%)': infra_scaling
        }
        
        title = "E-commerce Platform During Sales Event"
        explanation = """
        This simulation shows an e-commerce platform experiencing massive demand around day {:.0f}.
        The pattern indicates demand shock overwhelming capacity planning: transaction volume spikes 400%,
        cache hit ratio collapses causing thundering herd to backends, and infrastructure scaling hits limits.
        As systems struggle, payment gateway timeouts increase, session durations drop, and cart abandonment rises dramatically.
        This pattern is typical of Black Friday/Cyber Monday events without proper capacity planning.
        """
        
        return data, title, explanation.format(self.event_day)

    def simulate_cicd_pipeline_slowdown(self):
        """Simulate CI/CD pipeline slowdown due to technical debt"""
        # Build duration: steadily increasing
        build_time = 300 + 5 * self.time  # Gradual increase
        build_time = self.add_noise(build_time, 0.08)
        
        # Test execution time: exploding as tests accumulate
        test_time = 600 + 10 * self.time  # Baseline growth
        # Add accelerating growth due to test debt
        test_time[self.time > self.event_day-10] += 15 * ((self.time[self.time > self.event_day-10] - (self.event_day-10)) ** 1.5)
        test_time = self.add_noise(test_time, 0.1)
        
        # Repository size: growing steadily
        repo_size = 100 + 2 * self.time  # Linear growth
        repo_size = self.add_noise(repo_size, 0.03)
        
        # Static analysis time: increasing with codebase
        static_analysis = 150 + 3 * self.time  # Baseline growth
        # Add accelerating growth as complexity increases
        static_analysis[self.time > self.event_day-5] += 10 * ((self.time[self.time > self.event_day-5] - (self.event_day-5)) ** 1.2)
        static_analysis = self.add_noise(static_analysis, 0.07)
        
        # Deployment frequency: decreasing as pipeline slows
        deploy_freq = 10 - 0.15 * self.time  # Gradual decrease
        # Add sharper drop as delays compound
        deploy_freq[self.time > self.event_day-8] -= 0.2 * (self.time[self.time > self.event_day-8] - (self.event_day-8))
        deploy_freq = np.clip(deploy_freq, 0, 10)
        deploy_freq = self.add_noise(deploy_freq, 0.1)
        
        # PR wait times: increasing dramatically
        pr_wait = 4 + 0.1 * self.time  # Gradual increase
        # Add accelerating growth as bottlenecks form
        pr_wait[self.time > self.event_day-12] += 0.5 * ((self.time[self.time > self.event_day-12] - (self.event_day-12)) ** 1.3)
        pr_wait = self.add_noise(pr_wait, 0.15)
        
        data = {
            'Build Duration (sec)': build_time,
            'Test Execution Time (sec)': test_time,
            'Repository Size (MB)': repo_size,
            'Static Analysis Time (sec)': static_analysis,
            'Deployment Frequency (per day)': deploy_freq,
            'PR Wait Time (hours)': pr_wait
        }
        
        title = "CI CD Pipeline Slowdown Indicators"
        explanation = """
        This simulation shows a development pipeline slowdown becoming critical around day {:.0f}.
        The pattern indicates accumulating technical debt across the pipeline: test execution time
        growing exponentially, static analysis taking longer, build times increasing steadily,
        and deployment frequency dropping as a result. PR wait times increase dramatically,
        reducing developer productivity and creating a negative feedback loop.
        This pattern is common in maturing projects without dedicated optimization efforts.
        """
        
        return data, title, explanation.format(self.event_day)

    def simulate_security_incident(self):
        """Simulate data patterns during a security breach"""
        # Failed login attempts: suspicious patterns
        login_failures = 20 + 5 * np.sin(self.time / 2)  # Normal variations
        # Add targeted attack pattern
        attack_pattern = np.zeros_like(self.time)
        attack_pattern[self.time > self.event_day-5] = 40 * (1 - np.exp(-(self.time[self.time > self.event_day-5] - (self.event_day-5)) / 2))
        # Add periodic spikes (password spraying)
        for i in range(3):
            center = self.event_day-4+i*1.5
            attack_pattern[(self.time > center) & (self.time < center+0.3)] += 30
        login_failures = login_failures + attack_pattern
        login_failures = self.add_noise(login_failures, 0.2)
        
        # Outbound network traffic: data exfiltration
        outbound_traffic = 50 + 10 * np.sin(self.time / 3)  # Normal variations
        # Add suspicious outbound spikes
        exfil_pattern = np.zeros_like(self.time)
        for i in range(4):
            center = self.event_day+i*0.8
            exfil_pattern[(self.time > center) & (self.time < center+0.4)] += 40 * np.exp(-((self.time[(self.time > center) & (self.time < center+0.4)] - (center+0.2)) ** 2) / 0.01)
        outbound_traffic = outbound_traffic + exfil_pattern
        outbound_traffic = self.add_noise(outbound_traffic, 0.15)
        
        # Privileged operations: unusual increase
        privileged_ops = 10 + 2 * np.sin(self.time / 2)  # Normal variations
        # Add escalating privileged activities
        priv_increase = np.zeros_like(self.time)
        priv_increase[self.time > self.event_day] = 25 * (1 - np.exp(-(self.time[self.time > self.event_day] - self.event_day) / 3))
        privileged_ops = privileged_ops + priv_increase
        privileged_ops = self.add_noise(privileged_ops, 0.1)
        
        # After-hours activity: suspicious timing
        after_hours = 5 + 3 * np.sin(2 * np.pi * self.time / 1)  # Normal daily pattern
        # Add unusual after-hours activity
        unusual_pattern = np.zeros_like(self.time)
        for i in range(5):
            center = self.event_day-3+i*1.2
            # Activity during normal off-hours
            unusual_pattern[(self.time > center+0.4) & (self.time < center+0.7)] += 20
        after_hours = after_hours + unusual_pattern
        after_hours = self.add_noise(after_hours, 0.12)
        
        # Suspicious DNS queries: data exfiltration channel
        dns_queries = 30 + 5 * np.sin(self.time / 2)  # Normal patterns
        # Add suspicious DNS traffic
        dns_pattern = np.zeros_like(self.time)
        dns_pattern[self.time > self.event_day+1] = 45 * (1 - np.exp(-(self.time[self.time > self.event_day+1] - (self.event_day+1)) / 1.5))
        dns_queries = dns_queries + dns_pattern
        dns_queries = self.add_noise(dns_queries, 0.1)
        
        # Endpoint behavior changes: unusual processes
        endpoint_changes = 10 + 2 * np.random.random(self.n_points)  # Random baseline
        # Add malicious activity pattern
        malicious_pattern = np.zeros_like(self.time)
        malicious_pattern[self.time > self.event_day-1] = 30 * (1 - np.exp(-(self.time[self.time > self.event_day-1] - (self.event_day-1)) / 2))
        endpoint_changes = endpoint_changes + malicious_pattern
        endpoint_changes = self.add_noise(endpoint_changes, 0.15)
        
        data = {
            'Failed Login Attempts': login_failures,
            'Outbound Traffic (MB)': outbound_traffic,
            'Privileged Operations': privileged_ops,
            'After-hours Activity': after_hours,
            'Suspicious DNS Queries': dns_queries,
            'Endpoint Behavior Changes': endpoint_changes
        }
        
        title = "Security Incident Detection Indicators"
        explanation = """
        This simulation shows a sophisticated security breach starting around day {:.0f}.
        The pattern reveals an advanced persistent threat: initial access through password spraying,
        followed by privilege escalation, unusual after-hours activity, and data exfiltration through
        both direct outbound traffic and DNS tunneling. No single indicator is definitively malicious,
        but correlation across data sources reveals a clear attack pattern.
        This scenario demonstrates why security monitoring requires holistic analysis.
        """
        
        return data, title, explanation.format(self.event_day)

    def plot_scenario(self, data, title, explanation, plot_filename):
        """Plot the scenario data with explanation"""
        metrics = list(data.keys())
        
        plt.figure(figsize=(12, 10))
        gs = GridSpec(len(metrics), 1, height_ratios=[1] * len(metrics))
        
        for i, metric in enumerate(metrics):
            ax = plt.subplot(gs[i])
            ax.plot(self.dates, data[metric], 'k-', linewidth=1.5)
            
            # Add event line
            event_date = self.dates[int(self.event_day / self.days * self.n_points)]
            ax.axvline(x=event_date, color='r', linestyle='-', linewidth=1.5)
            
            ax.set_title(metric, loc='left', fontsize=10, fontweight='bold')
            
            # Format date axis
            if i == len(metrics) - 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                plt.xticks(rotation=45)
            else:
                ax.set_xticklabels([])
            
            # Add light background for readability
            ax.set_facecolor('#f9f9f9')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.figtext(0.5, 0.01, explanation, wrap=True, horizontalalignment='center', fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        print(f"{plot_filename}_plot.png")
        plt.savefig(f"{plot_filename}_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # return plt

# Function to demonstrate a single scenario
def show_scenario(scenario_name):
    """Show a specific scenario by name"""
    simulator = CrisisDataSimulator(n_points=500, days=30)
    
    # Dictionary mapping scenario names to functions
    scenarios = {
        'financial': simulator.simulate_financial_crisis,
        'sepsis': simulator.simulate_sepsis,
        'drought': simulator.simulate_drought,
        'crop_disease': simulator.simulate_crop_disease,
        'infrastructure': simulator.simulate_infrastructure_failure,
        'machine': simulator.simulate_machine_failure,
        'cloud': simulator.simulate_cloud_app_performance,
        'ecommerce': simulator.simulate_ecommerce_sales_event,
        'cicd': simulator.simulate_cicd_pipeline_slowdown,
        'security': simulator.simulate_security_incident
    }
    
    # Check if scenario exists
    if scenario_name not in scenarios:
        print(f"Scenario '{scenario_name}' not found. Available scenarios:")
        for name in scenarios.keys():
            print(f"- {name}")
        return
    
    # Generate and plot the scenario
    data, title, explanation = scenarios[scenario_name]()
    simulator.plot_scenario(data, title, explanation, os.path.join(os.path.dirname(os.path.abspath(__file__)),title.replace(" ", "_")))
    # plt.show()  # This will display the plot

# Run all scenarios in sequence
def show_all_scenarios():
    """Generate and display all Crisis scenarios"""
    simulator = CrisisDataSimulator(n_points=500, days=30)
    
    # Create all scenarios
    scenarios = [
        simulator.simulate_financial_crisis(),
        simulator.simulate_sepsis(),
        simulator.simulate_drought(),
        simulator.simulate_crop_disease(),
        simulator.simulate_infrastructure_failure(),
        simulator.simulate_machine_failure(),
        simulator.simulate_cloud_app_performance(),
        simulator.simulate_ecommerce_sales_event(),
        simulator.simulate_cicd_pipeline_slowdown(),
        simulator.simulate_security_incident()
    ]
    
    # Plot and show each scenario
    for data, title, explanation in scenarios:
        simulator.plot_scenario(data, title, explanation, os.path.join(os.path.dirname(os.path.abspath(__file__)),title.replace(" ", "_")))
        # plt.show()  # This will display each plot
        
# Interactive mode to select a scenario
def interactive_demo():
    print("Crisis Data Simulator - Interactive Mode")
    print("Available scenarios:")
    print("1. Financial Market Crisis")
    print("2. Patient Sepsis Development")
    print("3. Severe Drought Impact")
    print("4. Crop Disease Outbreak")
    print("5. Major Infrastructure Failure")
    print("6. Production Line Critical Failure")
    print("7. Cloud Application Performance Degradation")
    print("8. E-commerce Sales Event")
    print("9. CI/CD Pipeline Slowdown")
    print("10. Security Incident")
    print("11. Show All Scenarios")
    
    choice = input("\nEnter scenario number (1-11): ")
    
    if choice == '11':
        show_all_scenarios()
    elif choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']:
        scenario_map = {
            '1': 'financial',
            '2': 'sepsis',
            '3': 'drought',
            '4': 'crop_disease',
            '5': 'infrastructure',
            '6': 'machine',
            '7': 'cloud',
            '8': 'ecommerce',
            '9': 'cicd',
            '10': 'security'
        }
        show_scenario(scenario_map[choice])
    else:
        print("Invalid selection.")

# If running the script directly
if __name__ == "__main__":
    # Option 1: Direct function call to show a specific scenario
    # show_scenario('financial')
    
    # Option 2: Show all scenarios
    # show_all_scenarios()
    
    # Option 3: Interactive mode (recommended for exploration)
    #interactive_demo()

    simulator = CrisisDataSimulator()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generated_data_dir = os.path.join(base_dir, 'generated_data')
    os.makedirs(generated_data_dir, exist_ok=True)

    scenarios_to_run = [
        ('financial_crisis', simulator.simulate_financial_crisis),
        ('sepsis', simulator.simulate_sepsis),
        ('drought', simulator.simulate_drought),
        ('crop_disease', simulator.simulate_crop_disease),
        ('infrastructure_failure', simulator.simulate_infrastructure_failure),
        ('machine_failure', simulator.simulate_machine_failure),
        ('cloud_performance', simulator.simulate_cloud_app_performance),
        ('ecommerce', simulator.simulate_ecommerce_sales_event),
        ('cicd', simulator.simulate_cicd_pipeline_slowdown),
        ('security', simulator.simulate_security_incident)
    ]

     # Process and plot each scenario
    for name, simulation_func in scenarios_to_run:
        print(f"Processing scenario: {name}")
        
        # Generate data, title, and explanation
        data, title, explanation = simulation_func() # Gets all three return values
            
        # Define file paths
        plot_filename = os.path.join(generated_data_dir, 'plot', f"{name}_plot.png")
        data_filename = os.path.join(generated_data_dir, 'data', f"{name}_data.npz") # Npz file for data
        meta_filename = os.path.join(generated_data_dir, 'meta', f"{name}_meta.txt") # Text file for metadata

        # Plot the scenario
        simulator.plot_scenario(data, title, explanation, plot_filename)
        
        # --- Save the NumPy data ---
        try:
            np.savez_compressed(data_filename, **data) 
            print(f"Saved data for {name} to {data_filename}")
        except Exception as e:
            print(f"Error saving data for {name}: {e}")

        # --- Save the metadata (title and explanation) ---
        try:
            with open(meta_filename, 'w', encoding='utf-8') as f:
                f.write(f"Scenario Metadata: {data.keys()}\n")
                f.write(f"Title: {title}\n")
                f.write("-" * 20 + "\n") # Separator
                f.write("Explanation:\n")
                f.write(explanation) # Writes the already formatted explanation
            print(f"Saved metadata for {name} to {meta_filename}")
        except Exception as e:
            print(f"Error saving metadata for {name}: {e}")
                    
        print("\nAll scenarios processed and outputs saved.")