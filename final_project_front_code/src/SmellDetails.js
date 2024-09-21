import { h } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { route } from 'preact-router';
import Chart from 'chart.js/auto';
import ChartDataLabels from 'chartjs-plugin-datalabels'; // Import the datalabels plugin
import './SmellDetails.css';

function SmellDetails() {
  const chartRef = useRef(null); // Reference to the first chart canvas element
  const secondChartRef = useRef(null); // Reference to the second chart canvas element
  const barChartRef = useRef(null); // Reference to the bar chart canvas element
  const [smellName, setSmellName] = useState(''); // State to store smell name
  const [selectedOption, setSelectedOption] = useState(''); // State for dropdown selection
  const [secondPieChartData, setSecondPieChartData] = useState({}); // State for second pie chart data
  const [secondPieChartTitle, setSecondPieChartTitle] = useState(''); // State for second pie chart title
  const [barChartData, setBarChartData] = useState({}); // State to store data for bar chart
  const [uniquecounter, setUniqueCounter] = useState(); // State to store smell name

  const generateRandomColors = (num) => {
    const colors = [];
    for (let i = 0; i < num; i++) {
      const color = `hsl(${Math.random() * 360}, 70%, 70%)`; // Generate random HSL colors
      colors.push(color);
    }
    return colors;
  };

  // Define an array of 138 options
  const optionsList = ['alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal', 'anisic',
    'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy', 'bergamot',
    'berry', 'bitter', 'black currant', 'brandy', 'burnt', 'buttery', 'cabbage',
    'camphoreous', 'caramellic', 'cedar', 'celery', 'chamomile', 'cheesy',
    'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean', 'clove', 'cocoa',
    'coconut', 'coffee', 'cognac', 'cooked', 'cooling', 'cortex', 'coumarinic',
    'creamy', 'cucumber', 'dairy', 'dry', 'earthy', 'ethereal', 'fatty',
    'fermented', 'fishy', 'floral', 'fresh', 'fruit skin', 'fruity', 'garlic',
    'gassy', 'geranium', 'grape', 'grapefruit', 'grassy', 'green', 'hawthorn',
    'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth', 'jasmin', 'juicy',
    'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery', 'lemon', 'lily',
    'malty', 'meaty', 'medicinal', 'melon', 'metallic', 'milky', 'mint', 'muguet',
    'mushroom', 'musk', 'musty', 'natural', 'nutty', 'odorless', 'oily', 'onion',
    'orange', 'orangeflower', 'orris', 'ozone', 'peach', 'pear', 'phenolic',
    'pine', 'pineapple', 'plum', 'popcorn', 'potato', 'powdery', 'pungent',
    'radish', 'raspberry', 'ripe', 'roasted', 'rose', 'rummy', 'sandalwood',
    'savory', 'sharp', 'smoky', 'soapy', 'solvent', 'sour', 'spicy', 'strawberry',
    'sulfurous', 'sweaty', 'sweet', 'tea', 'terpenic', 'tobacco', 'tomato',
    'tropical', 'vanilla', 'vegetable', 'vetiver', 'violet', 'warm', 'waxy',
    'weedy', 'winey', 'woody'];

  // Function to handle user logout and navigate to the login page
  const handleLogout = () => {
    route('/login');
  };

  const newfetch = async (option) => {
    try {
      const response = await fetch('http://localhost:8000/get_pie_details');
      const data = await response.json();
      if (response.ok) {
        data.chart_data.forEach((sample) => {//gets the data for the pie chart 
          if (sample[0] === option) {// find right option 
            setSecondPieChartTitle("how many smells with " + option + " feature has 1 in "+option+" feature");
            setSecondPieChartData([(100 * (sample[1] / sample[2])).toFixed(2), (100 * (sample[2] - sample[1]) / sample[2]).toFixed(2)]);
            
          }
        });
      }
    } catch (error) {
      console.error("Error in fetch:", error);
    }
  };

  const fetchtosmell = async () => {
    try {
      const response = await fetch('http://localhost:8000/get_smell');
      const data = await response.json();
      if (response.ok) {

        setSmellName(data.smell_name);
        setBarChartData({
          labels: data.smell_name.split(";"),// gest labels for bar chart
          values: data.smell_data //define data for bar chart 
        });
        setUniqueCounter(data.unique)  //set  unique for sntence below title   
      }
    } catch (error) {
      console.error("Error in smell fetch:", error);
    }
  };

  useEffect(() => {
    fetchtosmell();
  }, []);


  useEffect(() => {
    if (Object.keys(secondPieChartData).length > 0) {
      initializeSecondPieChart();
    }
  }, [secondPieChartData]);

  useEffect(() => {
    if (Object.keys(barChartData).length > 0) {
      initializeBarChart();
    }
  }, [barChartData]);

 
  // Initialize the second pie chart after data is fetched
  const initializeSecondPieChart = () => {
    const ctx = secondChartRef.current.getContext('2d');
    if (ctx.chart) {
      ctx.chart.destroy();
    }
    const labels = Object.keys(["unique", "regular"]);
    const data = Object.values(secondPieChartData);
    const colors = generateRandomColors(labels.length);
    ctx.chart = new Chart(ctx, {
      type: 'pie',
      data: {
        labels: labels,
        datasets: [{
          data: data,
          backgroundColor: colors
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          datalabels: {
            color: 'white',
            font: { weight: 'bold', size: 16 },
            formatter: (value) => `${value}%`,
          },
          title: {
            display: true,
            text: secondPieChartTitle,
            font: { family: 'Raleway', size: 25, weight: 'bold' },
            color: '#000'
          }
        }
      },
      plugins: [ChartDataLabels]
    });
  };

  // Initialize the bar chart after data is fetched
  const initializeBarChart = () => {
    const ctx = barChartRef.current.getContext('2d');
    if (ctx.chart) {
      ctx.chart.destroy();
    }
    const labels = barChartData.labels;
    const data = barChartData.values;
    ctx.chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Feature Count',
          data: data,
          backgroundColor: generateRandomColors(10)
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: true },
          datalabels: {
            color: 'black',
            font: { weight: 'bold', size: 16 }
          },
          title: {
            display: true,           // Ensure the title is displayed
            text: `Bar Chart for ${selectedOption}`,  // Set dynamic title
            font: {
              family: 'Raleway',
              size: 30,              // Make the font size larger
              weight: 'bold'
            },
            color: '#000',           // Set title color to black
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Categories',
              font: { size: 16 }
            }
          },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Count',
              font: { size: 18 }
            },
            ticks: {
              stepSize: 1 // Ensure a step size of 1 to make small values like 1 appear larger
            }
          }
        },
        plugins: [ChartDataLabels]
      }
    });
  };
  

  const handleDropdownChange = (e) => {
    const option = e.target.value;
    setSelectedOption(option);
    newfetch(option);
  };
  return (
    h('div', { class: 'container' },
      h('div', { class: 'header' },
        h('div', { class: 'logo-container' },
          h('img', {
            src: 'src/LOGO.png',
            alt: 'Smelltify Logo',
            class: 'smelldetails-logo',
          })
        ),
        h('div', { class: 'title-container' },  // New container for title and uniqueLabel
          h('h1', { class: 'smelldetails-title' }, `Smell details: ${smellName}`),
          h('div', { class: 'smelldetails-unique-label', id: 'uniqueLabel' }, `you can create the smell ${smellName} in ${uniquecounter} unique ways`)
        ),
        h('button', {
          class: 'logout-button',
          onClick: handleLogout,
        }, 'Logout')
      ),
      h('div', { class: 'chart-container' },
        h('div', { class: 'bar-chart' },
          h('canvas', { id: 'barChart', ref: barChartRef }) // Canvas for the bar chart
        ),
        h('div', { class: 'dropdown-container', id: 'dropdown' },
          h('select', { value: selectedOption, onChange: handleDropdownChange, class: 'smelldetails-dropdown' },
            h('option', { value: '' }, 'Select feature'),
            optionsList.map((option, index) =>
              h('option', { value: option, key: index }, option)
            )
          )
        ),
        h('div', { class: 'second-pie-chart', id: 'pieChartID' },
          h('canvas', { id: 'secondPieChart', ref: secondChartRef, width: '6', height: '6' })
        )
      )
    )
  );
  
  
}

export default SmellDetails;
