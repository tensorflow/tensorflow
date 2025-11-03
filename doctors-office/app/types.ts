export interface Appointment {
  id: string;
  patientName: string;
  email: string;
  phone: string;
  dateOfBirth: string;
  appointmentDate: string;
  appointmentTime: string;
  doctor: string;
  reason: string;
  type: "new" | "follow-up";
  status: "scheduled" | "completed" | "cancelled";
  notes?: string;
  createdAt: string;
}

export interface Doctor {
  id: string;
  name: string;
  specialty: string;
}
