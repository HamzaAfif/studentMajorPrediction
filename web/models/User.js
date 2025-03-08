const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  firstName: { type: String, required: true },
  lastName: { type: String, required: true },
  school: { type: String, required: true },
  birthDate: { type: Date, required: true },
  notes: {
    math: Number,
    physics: Number,
    chemistry: Number,
    art: Number,
    economics: Number,
    preference: String,
  },
  prediction_report: { type: String },
  llmreply: { type: String },
  plots: { type: Object }, 
});

module.exports = mongoose.model('User', userSchema);
