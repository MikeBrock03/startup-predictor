import React, { useState } from 'react';
import axios from 'axios';
import Carousel from 'react-bootstrap/Carousel';
import 'bootstrap/dist/css/bootstrap.min.css';

const StartupForm = () => {

    const [isLoading, setIsLoading] = useState(false);
    const [predictionResult, setPredictionResult] = useState(null);

    const medianDates = {
        founded_at: new Date('2006-01-01'),
        first_funding_at: new Date('2007-09-01'),
        last_funding_at: new Date('2009-12-16'),
    };

    const [formData, setFormData] = useState({
        founded_at: '2015-12-15', // openai birthday awh
        first_funding_at: '', 
        last_funding_at: '', 
        age_first_funding_year: '',
        age_last_funding_year: '',
        age_first_milestone_year: '',
        age_last_milestone_year:'',
        funding_rounds: '',
        funding_total_usd: '',
        milestones: '',
        is_CA: false,
        is_NY: false,
        is_MA: false,
        is_TX: false,
        is_otherstate: false,
        is_software: false,
        is_web: false,
        is_mobile: false,
        is_enterprise: false,
        is_advertising: false,
        is_gamesvideo: false,
        is_ecommerce: false,
        is_biotech: false,
        is_consulting: false,
        is_othercategory: false,
        has_VC: false,
        has_angel: false,
        has_roundA: false,
        has_roundB: false,
        has_roundC: false,
        has_roundD: false,
        avg_participants: '',
        is_top500: false,
    });

    const [index, setIndex] = useState(0);
    const [showCarousel, setShowCarousel] = useState(false);
    const [hasRaisedFunding, setHasRaisedFunding] = useState(null);

    const convertDateToDaysFromMedian = (dateString, dateType) => {
        if (!dateString) return ''; // handle empty date strings
        const date = new Date(dateString);
        const medianDate = medianDates[dateType];
        return (date - medianDate) / (1000 * 3600 * 24); // Convert to days
    };

    const handleStart = () => {
        setShowCarousel(true);
    }

    const resetCategories = { 
        is_software: false,
        is_web: false,
        is_mobile: false,
        is_enterprise: false,
        is_advertising: false,
        is_gamesvideo: false,
        is_ecommerce: false,
        is_biotech: false,
        is_consulting: false,
        is_othercategory: false,
    };

    const resetStates = { 
        is_CA: false,
        is_NY: false,
        is_MA: false,
        is_TX: false,
        is_otherstate: false,
    };

    const resetRounds = { 
        has_roundA: false,
        has_roundB: false,
        has_roundC: false,
        has_roundD: false,
    };

    const totalItems = 15;

    const calculateAges = (foundedAt, firstFundingAt, lastFundingAt) => {
        const getYear = (date) => date ? new Date(date).getFullYear() : null;
        const foundedYear = getYear(foundedAt);
        const firstFundingYear = getYear(firstFundingAt);
        const lastFundingYear = getYear(lastFundingAt);
    
        // Calculate the first funding gap
        let firstFundingGap = firstFundingYear ? firstFundingYear - foundedYear : 0;
        // Set to a large number if the gap is zero
        firstFundingGap = firstFundingGap === 0 ? 1000000000 : firstFundingGap;
    
        return {
            age_first_funding_year: firstFundingYear ? firstFundingYear - foundedYear : 0,
            age_last_funding_year: lastFundingYear ? lastFundingYear - foundedYear : 0,
            first_funding_gap: firstFundingGap,
        };
    };

    const handleChange = (e) => {
        const { name, value, type, checked, id } = e.target;
        let newFormData = { ...formData };
        
        // Check if the input type is 'number' and convert the value to a number
        if (type === 'number') {
            // Convert number input values to numbers (or default to 0 if empty)
            newFormData[name] = value ? Number(value) : '';
        } else if (type === 'checkbox') {
            // Handle checkbox inputs
            newFormData[name] = checked;
        } else if (type === 'radio' && (name !== 'category') && (name !== 'state')) {
            // Convert radio input values to boolean
            newFormData[name] = value === 'true';
        } 
       

        
        
        if (name === 'category' || name === 'state') {
            console.log('when going into the check', newFormData);
            const resetObject = name === 'category' ? resetCategories : resetStates;
            console.log('after reset object', newFormData);

            // Reset all related fields to false
            Object.keys(resetObject).forEach(key => {
                console.log('reset object is', resetObject);
                console.log('key is', key);
                newFormData[key] = false;
            });
            console.log('after reset fields', newFormData);

            // Set the selected one to true
            console.log('id is', id);
            newFormData[id] = true;
            console.log('after setting id to true', newFormData);

        }
        // Handle funding rounds
        else if (Object.keys(resetRounds).includes(name)) {
            const roundValue = checked;
            // Update the selected round
            newFormData[name] = roundValue;
            // Adjust other rounds based on the selected one
            const roundOrder = ['has_roundA', 'has_roundB', 'has_roundC', 'has_roundD'];
            const roundIndex = roundOrder.indexOf(name);
            roundOrder.forEach((round, index) => {
                if (roundValue && index <= roundIndex) {
                    newFormData[round] = true;
                } else if (!roundValue && index >= roundIndex) {
                    newFormData[round] = false;
                }
            });
        }
    
        if (['founded_at', 'first_funding_at', 'last_funding_at'].includes(name)) {
            newFormData[name] = value;
            const ages = calculateAges(newFormData.founded_at, newFormData.first_funding_at, newFormData.last_funding_at);
            newFormData = { ...newFormData, ...ages };
        }

        if (['has_roundA', 'has_roundB', 'has_roundC', 'has_roundD'].includes(name)) {
            const fundingRounds = ['has_roundA', 'has_roundB', 'has_roundC', 'has_roundD']
                .filter(round => newFormData[round])
                .length;
            newFormData.funding_rounds = fundingRounds;
        }
        // Handle other inputs
        setFormData(newFormData);

        if (type === 'radio' && index < totalItems - 1) {
            setIndex(index + 1);
        }
        console.log('newFormData', newFormData);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            e.preventDefault(); // Prevent form submission if in a form
            setIndex(index + 1);
        }
    };
    

    const handleSubmit = async (e) => {
        e.preventDefault(); // Prevent the default form submission
        console.log('form data is', formData);
        console.log('ok we about to submit')

        const processedFormData = {
            ...formData,
            founded_at: convertDateToDaysFromMedian(formData.founded_at, 'founded_at'),
            first_funding_at: convertDateToDaysFromMedian(formData.first_funding_at, 'first_funding_at'),
            last_funding_at: convertDateToDaysFromMedian(formData.last_funding_at, 'last_funding_at'),
            // Process other date fields similarly
        };

        console.log('processed form data is', processedFormData);

        try {
            console.log('ok we ab to get da response');
            const response = await fetch('https://startup-predictor-810e3a7fc484.herokuapp.com/predict/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(processedFormData),
            });
            console.log('response is', response);

    
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
    
            const result = await response.json();
            console.log('Prediction result:', result);
            setPredictionResult(result.probability[0]); // Store the prediction result
            setIsLoading(false); // Set loading to false
        } catch (error) {
            console.error('There was an error!', error);
            setIsLoading(false); // Set loading to false
        }
    };

    const handleUseFirstFundingDate = () => {
        setFormData({ ...formData, last_funding_at: formData.first_funding_at });      
        setIndex(index + 1);
    };

    const handleMilestoneChange = (e) => {
        const milestoneChecked = e.target.checked;
        const milestoneName = e.target.name;
    
        // Update the individual milestone in a local copy of formData
        const updatedFormData = { ...formData, [milestoneName]: milestoneChecked };
    
        // Calculate the total number of milestones achieved
        const milestoneKeys = [
            'milestone_product_launch',
            'milestone_customers',
            'milestone_revenue',
            'milestone_patent',
            'milestone_funding_round',
            'milestone_market_expansion',
            'milestone_award',
            'milestone_partnership'
        ];
        const totalMilestones = milestoneKeys.reduce((total, key) => {
            return total + (updatedFormData[key] ? 1 : 0);
        }, 0);
    
        // Update the milestones count in formData
        setFormData({ ...formData, milestones: totalMilestones });
    };

    const handleSelect = (selectedIndex, e) => {
        setIndex(selectedIndex);
    };

    const handleNext = () => {
        if (index < totalItems - 1) {
            setIndex(index + 1);
        }
    };

    const handlePrev = () => {
        if (index > 0) {
            setIndex(index - 1);
        }
    };

    return (
        <div className="form-container">
        {!showCarousel && (
            <>
                <h1 style={{ marginBottom: '20px', fontWeight: 'bold' }}>we use AI to tell you if your startup is going to pan out.</h1>
                <p style={{ marginBottom: '50px' }}>extended from work in CS221 at Stanford University.</p>
            
                <p style={{ marginBottom: '10px' }}>
                    <a className="peachpuff-link" href="https://docs.google.com/document/d/1uq-6tsF-ae2oElJvrMVzB2EFo5LI0sHGhWPaSr4gEfc/edit?usp=sharing" target="_blank" rel="noopener noreferrer"> paper </a> | 
                    <a className="peachpuff-link" href="https://www.youtube.com/watch?v=kkI3YXysFrc" target="_blank" rel="noopener noreferrer"> video explanation </a> | 
                    <a className="peachpuff-link" href="https://github.com/ninaboord/startup-predictor" target="_blank" rel="noopener noreferrer"> github </a> | 
                    <a className="peachpuff-link" href="https://www.kaggle.com/datasets/manishkc06/startup-success-prediction/data" target="_blank" rel="noopener noreferrer"> dataset </a>
                </p>
            
                <p style={{ marginBottom: '20px' }}>authors: 
                    <a className="peachpuff-link" href="https://www.linkedin.com/in/michael-brockman-15b5b916a/" target="_blank" rel="noopener noreferrer"> Michael Brockman</a>, 
                    <a className="peachpuff-link" href="https://www.linkedin.com/in/hamed-hekmat/" target="_blank" rel="noopener noreferrer"> Hamed Hekmat</a>, 
                    <a className="peachpuff-link" href="https://www.linkedin.com/in/nina-boord/" target="_blank" rel="noopener noreferrer"> Nina Boord</a>, 
                    <a className="peachpuff-link" href="https://www.linkedin.com/in/iristfu/" target="_blank" rel="noopener noreferrer"> Iris Fu</a>
                </p>
            
                <p style={{ marginBottom: '20px' }}>for fun, not business advice!</p>
            
                <button onClick={handleStart} className="start-button">Start</button>
            </>
        )}
        {isLoading && (
            <div>Loading...</div> // Loading screen content
        )}

        {!isLoading && predictionResult !== null && (
            <div className={predictionResult > 50 ? "celebratory-screen" : "sad-screen"}>
                {predictionResult > 50 ? (
                    <div>
                        {/* Celebration message or image */}
                        <h1>Congratulations! Your startup seems promising!</h1>
                        <p>This startup is over 50% likelihood of succeeding, based on our prior data.</p>
                    </div>
                ) : (
                    <div>
                        {/* Sad message or image */}
                        <label>Maybe this isn't the winner - yet!</label>
                        <p>This startup has less than 50% likelihood of succeeding, based on our prior data.</p>
                    </div>
                )}
                <button onClick={() => window.location.reload()} className="try-again-button">Try Again</button>
            </div>
        )}

            {showCarousel && !isLoading && predictionResult === null && (
                <form onSubmit={handleSubmit}>
                    <Carousel activeIndex={index} indicators={false} controls={false}>

                    <Carousel.Item>
                        <div className="carousel-form-item">
                            <label>When were you founded?</label>
                            <input type="date" name="founded_at" value={formData.founded_at} onChange={handleChange} onKeyDown={handleKeyDown} />
                        </div>
                    </Carousel.Item>
                    <Carousel.Item>
                        <div className="carousel-form-item">
                            <label>What state is your startup based in?</label>
                            <div className="radio-buttons-container">
                                <div className="radio-group">
                                    <input type="radio" id="is_CA" name="state" value="is_CA" checked={formData.is_CA} onChange={handleChange}  />
                                    <label htmlFor="is_CA">California</label>
                                </div>
                                <div className="radio-group">
                                    <input type="radio" id="is_NY" name="state" value="is_NY" checked={formData.is_NY} onChange={handleChange} />
                                    <label htmlFor="is_NY">New York</label>
                                </div>
                                <div className="radio-group">
                                    <input type="radio" id="is_MA" name="state" value="is_MA" checked={formData.is_MA} onChange={handleChange} />
                                    <label htmlFor="is_MA">Massachusetts</label>
                                </div>
                                <div className="radio-group">
                                    <input type="radio" id="is_TX" name="state" value="is_TX" checked={formData.is_TX} onChange={handleChange} />
                                    <label htmlFor="is_TX">Texas</label>
                                </div>
                                <div className="radio-group">
                                    <input type="radio" id="is_otherstate" name="state" value="is_otherstate" checked={formData.is_otherstate} onChange={handleChange} />
                                    <label htmlFor="is_otherstate">Other</label>
                                </div>
                            </div>
                        </div>
                    </Carousel.Item>

                    <Carousel.Item>
                        <div className="carousel-form-item">
                            <label>What category best describes your startup?</label>
                                <div className="radio-buttons-container">
                                <div className="radio-group">
                                    <input type="radio" id="is_software" name="category" value="true" checked={formData.is_software} onChange={handleChange} />
                                    <label htmlFor="is_software">Software</label>
                                </div>
                                <div className="radio-group">
                                    <input type="radio" id="is_web" name="category" value="true" checked={formData.is_web} onChange={handleChange} />
                                    <label htmlFor="is_web">Web</label>
                                </div>
                                <div className="radio-group">
                                    <input type="radio" id="is_mobile" name="category" value="true" checked={formData.is_mobile} onChange={handleChange} />
                                    <label htmlFor="is_mobile">Mobile</label>
                                </div>
                                <div className="radio-group">
                                    <input type="radio" id="is_enterprise" name="category" value="true" checked={formData.is_enterprise} onChange={handleChange} />
                                    <label htmlFor="is_enterprise">Enterprise</label>
                                </div>
                                <div className="radio-group">
                                    <input type="radio" id="is_advertising" name="category" value="true" checked={formData.is_advertising} onChange={handleChange} />
                                    <label htmlFor="is_advertising">Advertising</label>
                                </div>
                                <div className="radio-group">
                                    <input type="radio" id="is_gamesvideo" name="category" value="true" checked={formData.is_gamesvideo} onChange={handleChange} />
                                    <label htmlFor="is_gamesvideo">Games/Video</label>
                                </div>
                                <div className="radio-group">
                                    <input type="radio" id="is_ecommerce" name="category" value="true" checked={formData.is_ecommerce} onChange={handleChange} />
                                    <label htmlFor="is_ecommerce">Ecommerce</label>
                                </div>
                                <div className="radio-group">
                                    <input type="radio" id="is_biotech" name="category" value="true" checked={formData.is_biotech} onChange={handleChange} />
                                    <label htmlFor="is_biotech">Biotech</label>
                                </div>
                                <div className="radio-group">
                                    <input type="radio" id="is_consulting" name="category" value="true" checked={formData.is_consulting} onChange={handleChange} />
                                    <label htmlFor="is_consulting">Consulting</label>
                                </div>
                                <div className="radio-group">
                                    <input type="radio" id="is_othercategory" name="category" value="true" checked={formData.is_othercategory} onChange={handleChange} />
                                    <label htmlFor="is_othercategory">Other</label>
                                </div>
                                
                        </div>
                        </div>
                    </Carousel.Item>

                    <Carousel.Item>
                        <div className="carousel-form-item">
                            <label>How many founders do you have?</label>
                            <input 
                                type="number"
                                name="avg_participants" 
                                value={formData.avg_participants}
                                onChange={handleChange} 
                                onKeyDown={handleKeyDown}
                                min="1"
                                step="1"
                            />
                        </div>
                    </Carousel.Item>

                    <Carousel.Item>
                        <div className="carousel-form-item">
                            <label>Are you VC backed?</label>
                            <div className="radio-buttons-container">
                            <div className="radio-group">
                                <input type="radio" id="vc_yes" name="has_VC" value="true" checked={formData.has_VC === true} onChange={handleChange} />
                                <label htmlFor="vc_yes">Yes</label>
                            </div>
                            <div className="radio-group">
                                <input type="radio" id="vc_no" name="has_VC" value="false" checked={formData.has_VC === false} onChange={handleChange} />
                                <label htmlFor="vc_no">No</label>
                            </div>
                            </div>
                        </div>
                    </Carousel.Item>

                    <Carousel.Item>
                        <div className="carousel-form-item">
                            <label>Do you have an angel investor?</label>
                            <div className="radio-buttons-container">
                            <div className="radio-group">
                                <input type="radio" id="angel_yes" name="has_angel" value="true" checked={formData.has_angel === true} onChange={handleChange} />
                                <label htmlFor="angel_yes">Yes</label>
                            </div>
                            <div className="radio-group">
                                <input type="radio" id="angel_no" name="has_angel" value="false" checked={formData.has_angel === false} onChange={handleChange} />
                                <label htmlFor="angel_no">No</label>
                            </div>
                            </div>
                        </div>
                    </Carousel.Item>
                          
                    <Carousel.Item>
                        <div className="carousel-form-item">
                            <label>When did you first raise funding?</label>
                            <input type="date" name="first_funding_at" value={formData.first_funding_at} onChange={handleChange} onKeyDown={handleKeyDown}/>
                            <p>If you have never raised funding, just click 'Next'.</p>
                        </div>
                    </Carousel.Item>
                    
                    <Carousel.Item>
                        <div className="carousel-form-item">
                            <label>When did you last raise funding?</label>
                            <input type="date" name="last_funding_at" value={formData.last_funding_at} onChange={handleChange} onKeyDown={handleKeyDown}/>
                            {formData.first_funding_at && (
                                <button 
                                    type="button" 
                                    onClick={handleUseFirstFundingDate}
                                    className="use-same-date-button"
                                >
                                    Use Date from First Funding
                                </button>
                            )}
                            <p>If you have never received funding, just click 'Next'.</p>
                        </div>
                    </Carousel.Item>

                    <Carousel.Item>
                        <div className="carousel-form-item">
                            <label>What is the highest round of funding you have completed?</label>
                            <div className='checkbox-buttons-container'>
                                <div className='checkbox-group'>
                                    <input type="checkbox" id="none" name="no_funding" checked={!formData.has_roundA && !formData.has_roundB && !formData.has_roundC && !formData.has_roundD} onChange={() => setFormData({ ...formData, has_roundA: false, has_roundB: false, has_roundC: false, has_roundD: false })} />
                                    <label htmlFor="none">None</label>
                                </div>
                                <div className='checkbox-group'>
                                    <input type="checkbox" id="round_a" name="has_roundA" checked={formData.has_roundA} onChange={handleChange} />
                                    <label htmlFor="round_a">Round A or Pre-seed</label>
                                </div>
                                <div className='checkbox-group'>
                                    <input type="checkbox" id="round_b" name="has_roundB" checked={formData.has_roundB} onChange={handleChange} />
                                    <label htmlFor="round_b">Round B</label>
                                </div>
                                <div className='checkbox-group'>
                                    <input type="checkbox" id="round_c" name="has_roundC" checked={formData.has_roundC} onChange={handleChange} />
                                    <label htmlFor="round_c">Round C</label>
                                </div>
                                <div className='checkbox-group'>
                                    <input type="checkbox" id="round_d" name="has_roundD" checked={formData.has_roundD} onChange={handleChange} />
                                    <label htmlFor="round_d">Round D</label>
                                </div>   
                            </div>
                        </div>
                    </Carousel.Item>
                    
                    <Carousel.Item>
                        <div className="carousel-form-item">
                            <label>How much have you raised (in $ dollars)?</label>
                            <input 
                                type="number" 
                                name="funding_total_usd" 
                                value={formData.funding_total_usd} 
                                onChange={handleChange} 
                                onKeyDown={handleKeyDown}
                                min="0" 
                                step="any" // Allows for decimal numbers
                            />
                            <p>This can include personal investments if you have never officially raised.</p>
                        </div>
                    </Carousel.Item>

                    <Carousel.Item>
                        <div className="carousel-form-item">
                            <label>Select the milestones your startup has achieved:</label>
                            <div className='checkbox-buttons-container'>
                                <div className='checkbox-group'>
                                    <input type="checkbox" id="product_launch" name="milestone_product_launch" onChange={handleMilestoneChange} />
                                    <label htmlFor="product_launch">Launched product</label>
                                </div>
                                <div className='checkbox-group'>
                                    <input type="checkbox" id="customers" name="milestone_customers" onChange={handleMilestoneChange} />
                                    <label htmlFor="customers">Secured goal number of customers</label>
                                </div>
                                <div className='checkbox-group'>
                                    <input type="checkbox" id="revenue" name="milestone_revenue" onChange={handleMilestoneChange} />
                                    <label htmlFor="revenue">Reached revenue target</label>
                                </div>
                                <div className='checkbox-group'>
                                    <input type="checkbox" id="patent" name="milestone_patent" onChange={handleMilestoneChange} />
                                    <label htmlFor="patent">Secured intellectual property</label>
                                </div>
                                <div className='checkbox-group'>
                                    <input type="checkbox" id="funding_round" name="milestone_funding_round" onChange={handleMilestoneChange} />
                                    <label htmlFor="funding_round">Completed funding round</label>
                                </div>
                                <div className='checkbox-group'>
                                    <input type="checkbox" id="market_expansion" name="milestone_market_expansion" onChange={handleMilestoneChange} />
                                    <label htmlFor="market_expansion">Expanded to a new market</label>
                                </div>
                                <div className='checkbox-group'>
                                    <input type="checkbox" id="award" name="milestone_award" onChange={handleMilestoneChange} />
                                    <label htmlFor="award">Won an award</label>
                                </div>
                                <div className='checkbox-group'>
                                    <input type="checkbox" id="partnership" name="milestone_partnership" onChange={handleMilestoneChange} />
                                    <label htmlFor="partnership">Partnered with a notable company</label>
                                </div>
                            </div>
                        </div>
                    </Carousel.Item>

                    
                    <Carousel.Item>
                        <div className="carousel-form-item">
                            <label>How old was the company when you hit your first milestone?</label>
                            <input type="number" min="0" step="1" name="age_first_milestone_year" value={formData.age_first_milestone_year} onChange={handleChange} onKeyDown={handleKeyDown}/>
                            <p>For example, when you launched your first product or secured a patent.</p>
                            <p>(in years)</p>

                        </div>
                    </Carousel.Item>
                    <Carousel.Item>
                        <div className="carousel-form-item">
                            <label>How old was the company when you hit your most recent milestone?</label>
                            <input type="number" min="0" step="1" name="age_last_milestone_year" value={formData.age_last_milestone_year} onChange={handleChange} onKeyDown={handleKeyDown}/>
                            <p>For example, when you made a revenue goal or expanded to a new market.</p>

                        </div>
                    </Carousel.Item>
        
                    
                    <Carousel.Item>
                        <div className="carousel-form-item">
                            <label>Do you happen to be in the Fortune 500 or Forbes Global 2000?</label>
                            <div className="radio-buttons-container">
                                <div className="radio-group">
                                    <input type="radio" id="top500_yes" name="is_top500" value="true" checked={formData.is_top500 === true} onChange={handleChange} />
                                    <label htmlFor="top500_yes">Yes</label>
                                </div>
                                <div className="radio-group">
                                    <input type="radio" id="top500_no" name="is_top500" value="false" checked={formData.is_top500 === false} onChange={handleChange} />
                                    <label htmlFor="top500_no">No</label>
                                </div>
                            </div>
                        </div>
                    </Carousel.Item>
                    </Carousel>
                    <div className="carousel-controls">
                        <button type="button" onClick={() => setIndex(index - 1)} disabled={index === 0}>Prev</button>
                        <span style={{ margin: '0 10px' }}></span> {/* This span adds space between the buttons */}
                        <button 
                            type={index === totalItems - 1 ? "submit" : "button"}
                            onClick={() => index === totalItems - 1 ? handleSubmit() : setIndex(index + 1)}
                        >
                            {index === totalItems - 1 ? "Judge My Startup" : "Next"}
                        </button>
                    </div>
                    
                </form>
            )}
        </div>
    );
};

export default StartupForm;