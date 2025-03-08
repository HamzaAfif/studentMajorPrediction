const express = require('express');
const mongoose = require('mongoose');
const session = require('express-session');
const MongoDBStore = require('connect-mongodb-session')(session);
const bcrypt = require('bcrypt');
const User = require('./models/User');
const axios = require('axios');
const http = require('http');
const { Server } = require('socket.io');
const { io: pythonSocketClient } = require('socket.io-client');
const cors = require('cors'); 

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: 'http://localhost:3000', // Allow front-end origin
    methods: ['GET', 'POST'],       // Allow specific HTTP methods
    credentials: true               // Allow credentials (e.g., cookies, auth headers)
  }
});

const pythonSocket = pythonSocketClient('http://127.0.0.1:5000');


const MONGO_URI = 'mongodb://localhost:27017/simpleAuth';
mongoose
  .connect(MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Connected to MongoDB'))
  .catch((err) => console.log('MongoDB connection error:', err));


const store = new MongoDBStore({
  uri: MONGO_URI,
  collection: 'sessions',
});


app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(cors({ origin: 'http://localhost:3000', credentials: true })); // Use CORS middleware
app.use(
  session({
    secret: 'my_secret_key',
    resave: false,
    saveUninitialized: false,
    store: store,
  })
);


app.set('view engine', 'ejs');


app.get('/', (req, res) => {
  if (req.session.userId) {
    return res.redirect('/home');
  }
  res.render('login');
});

app.post('/login', async (req, res) => {
  const { email, password } = req.body;
  const user = await User.findOne({ email });

  if (user && (await bcrypt.compare(password, user.password))) {
    req.session.userId = user._id;
    return res.redirect('/home');
  }

  res.send('Invalid email or password.');
});

app.get('/register', (req, res) => {
  res.render('register');
});

app.post('/register', async (req, res) => {
  const { email, password, firstName, lastName, school, birthDate } = req.body;

  const hashedPassword = await bcrypt.hash(password, 10);

  const newUser = new User({
    email,
    password: hashedPassword,
    firstName,
    lastName,
    school,
    birthDate
  });

  await newUser.save();
  res.redirect('/');
});


app.use('/public', express.static('public'));

app.get('/home', async (req, res) => {
    if (!req.session.userId) {
      return res.redirect('/');
    }
  
    try {
      const user = await User.findById(req.session.userId);
  
      if (!user) {
        return res.redirect('/');
      }
  
      const notes = user.notes || {};
      const predictionReport = user.prediction_report || null;
      const llmReply = user.llmreply || null;
  
      const plots = {};

      if (user.plots && typeof user.plots === 'object') {
        Object.entries(user.plots).forEach(([key, value]) => {
          if (typeof value === 'string') {
            plots[key] = value.replace(/^web\\/, '/').replace(/\\/g, '/');
          } else {
            console.warn(`Unexpected plot value for key ${key}:`, value);
          }
        });
      }
  
      console.log('Normalized Plots for Frontend:', plots);
  
      res.render('home', { message: 'Welcome!', notes, predictionReport, llmReply, plots, user });
    } catch (error) {
      console.error('Error fetching user data:', error);
      res.redirect('/');
    }
  });
  
  
  app.post('/update-account', async (req, res) => {
    if (!req.session.userId) {
      return res.redirect('/');
    }
  
    try {
      const { firstName, lastName, school, birthDate } = req.body;
  
      // Find the user by their ID and update their details
      const updatedUser = await User.findByIdAndUpdate(
        req.session.userId,
        { firstName, lastName, school, birthDate },
        { new: true }
      );
  
      if (!updatedUser) {
        return res.redirect('/');
      }
  
      // Redirect back to the home page with the updated data
      res.redirect('/home');
    } catch (error) {
      console.error('Error updating user data:', error);
      res.redirect('/home');
    }
  });
  
  


app.post('/logout', (req, res) => {
  req.session.destroy((err) => {
    if (err) {
      return res.send('Error logging out.');
    }
    res.redirect('/');
  });
});

// Real-time updates from Python
pythonSocket.on('connect', () => {
  console.log('Connected to Python WebSocket server');
});

pythonSocket.on('update', (data) => {
    console.log('Real-time update from Python:', data.message);
    io.emit('update', { message: data.message }); 
});

io.on('connection', (socket) => {
    console.log('A front-end client connected');

    socket.on('disconnect', () => {
        console.log('A front-end client disconnected');
    });
});


pythonSocket.on('disconnect', () => {
  console.log('Disconnected from Python WebSocket server');
});



app.post('/trigger-predict', async (req, res) => {
    try {
        const response = await axios.get('http://127.0.0.1:5000/api/notes'); // Call Python API
        console.log('Prediction response:', response.data);

        io.emit('update', { message: response.data.predicted_major });

        res.redirect('/home');
    } catch (error) {
        console.error('Error calling Python API:', error);

        io.emit('update', { message: 'Error fetching prediction from Python API.' });
        res.redirect('/home');
    }
});

// Send data to Python API
app.post('/send-to-python', async (req, res) => {
  const { inputText } = req.body;

  try {
    await axios.post('http://127.0.0.1:5000/api/receive', { text: inputText });

    console.log('Data sent to Python server:', inputText);

    res.redirect('/home');
  } catch (error) {
    console.error('Error sending data to Python API:', error);
    res.send('Failed to send data to Python API.');
  }
});


app.post('/submit-notes', async (req, res) => {
    if (!req.session.userId) {
        return res.status(401).json({ error: 'Unauthorized' });
    }

    const { math, physics, chemistry, art, economics, preference } = req.body;

    try {
        const pythonResponse = await axios.post('http://127.0.0.1:5000/api/notes', {
            math,
            physics,
            chemistry,
            art,
            economics,
            preference,
            userId: req.session.userId,
        });

        console.log('Response from Python back-end:', pythonResponse.data);

        const predictedMajor = pythonResponse.data.predicted_major;

        const updatedUser = await User.findByIdAndUpdate(
            req.session.userId,
            {
                $set: {
                    notes: {
                        math: Number(math),
                        physics: Number(physics),
                        chemistry: Number(chemistry),
                        art: Number(art),
                        economics: Number(economics),
                        preference,
                    },
                },
            },
            { new: true } // Return the updated document
        );

        if (!updatedUser) {
            return res.status(404).json({ error: 'User not found' });
        }

        
        res.status(200).json({
            message: 'Notes saved successfully and forwarded to Python back-end',
            user: updatedUser,
            predicted_major: predictedMajor, 
        });

        

    } catch (error) {
        console.error('Error in /submit-notes:', error);
        res.status(500).json({ error: 'Failed to process notes' });
    }
});


const PORT = 3000;
server.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
