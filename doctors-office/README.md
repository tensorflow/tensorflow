# Doctor's Office Appointment System

A modern, responsive web application for managing doctor's office appointments and follow-ups.

## Features

### 📅 Appointment Booking
- Book new appointments with comprehensive patient information
- Select from multiple doctors and specialties
- Choose from available time slots
- Specify appointment type (new or follow-up)
- Add detailed reason for visit and additional notes
- Real-time form validation

### 📋 Appointment Management
- View all appointments in one place
- Filter appointments by status (scheduled, completed, cancelled)
- Search appointments by patient name, doctor, or reason
- Mark appointments as completed
- Cancel appointments with confirmation
- Schedule follow-up visits

### 📊 Dashboard
- View appointment statistics
- Track scheduled, completed, and cancelled appointments
- Easy navigation between features

## Technology Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State Management**: React Hooks (useState, useEffect)
- **Data Persistence**: Browser LocalStorage

## Getting Started

### Prerequisites
- Node.js 18+ installed
- npm or yarn package manager

### Installation

1. Navigate to the project directory:
```bash
cd doctors-office
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

### Build for Production

```bash
npm run build
```

This creates an optimized production build in the `.next` folder.

## Project Structure

```
doctors-office/
├── app/
│   ├── appointments/       # Appointments list and management
│   │   └── page.tsx
│   ├── book/              # Appointment booking form
│   │   └── page.tsx
│   ├── components/        # Reusable components
│   │   └── Navigation.tsx
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   ├── page.tsx          # Home page
│   └── types.ts          # TypeScript type definitions
├── public/               # Static assets
├── next.config.ts       # Next.js configuration
├── tailwind.config.ts   # Tailwind CSS configuration
├── tsconfig.json        # TypeScript configuration
└── package.json         # Project dependencies
```

## Features in Detail

### Appointment Booking Form
- **Patient Information**: Name, email, phone, date of birth
- **Appointment Details**: Date, time, doctor selection
- **Visit Information**: Reason for visit, appointment type, additional notes
- **Validation**: Comprehensive form validation with error messages
- **Date Restrictions**: Prevents booking appointments in the past

### Appointments Page
- **List View**: All appointments with detailed information
- **Status Management**: Update appointment status (scheduled, completed, cancelled)
- **Search & Filter**: Find appointments quickly
- **Follow-up Scheduling**: Easy follow-up appointment booking
- **Statistics Dashboard**: Visual overview of appointment metrics

### Doctors Available
1. Dr. Sarah Johnson - General Practice
2. Dr. Michael Chen - Pediatrics
3. Dr. Emily Rodriguez - Cardiology
4. Dr. James Wilson - Dermatology

### Available Time Slots
- Morning: 9:00 AM - 11:30 AM (30-minute intervals)
- Afternoon: 2:00 PM - 4:30 PM (30-minute intervals)

## Data Storage

The application uses browser LocalStorage to persist appointment data. This means:
- Data is stored locally in the user's browser
- Data persists across browser sessions
- Data is specific to each browser/device
- No backend server required for basic functionality

## Future Enhancements

Potential features for future development:
- Backend API integration for multi-user support
- Email/SMS appointment reminders
- Doctor availability calendar
- Patient medical history
- Prescription management
- Payment processing
- Admin dashboard for office staff
- Real-time appointment availability
- Video consultation integration

## Browser Compatibility

The application works on all modern browsers:
- Chrome (recommended)
- Firefox
- Safari
- Edge

## Responsive Design

The application is fully responsive and works on:
- Desktop computers
- Tablets
- Mobile phones

## License

This project is open source and available for educational and commercial use.

## Support

For issues or questions, please refer to the project documentation or contact the development team.
